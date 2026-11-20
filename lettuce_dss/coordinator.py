"""Shared-message coordinator with parallel context retrieval and bounded feedback."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy
import hashlib
import json
import tempfile
import time
import uuid
from lettuce_dss.core import utc_now
from lettuce_dss import __version__
from lettuce_dss.agents.reporting import ReportingAgent

def input_fingerprint(image_bytes: bytes, farm: dict, model_mode: str, device: str) -> str:
    h=hashlib.sha256(image_bytes)
    h.update(json.dumps({'farm':farm,'mode':model_mode,'device':device},sort_keys=True,default=str).encode())
    return h.hexdigest()

class Coordinator:
    def __init__(self, project_root: Path, diagnosis_agent=None, environment_agent=None,
                 recommendation_agent=None, quality_agent=None, reporting_agent=None):
        self.root=Path(project_root)
        config_path = self.root / 'config/app.json'
        settings = json.loads(config_path.read_text()) if config_path.is_file() else {}
        if environment_agent is None:
            from lettuce_dss.agents.environment import EnvironmentalAgent
            environment_agent=EnvironmentalAgent(self.root/'runtime'/'weather_cache',project_root=self.root,
                timeout=settings.get('weather_timeout_seconds', 5.0))
        if diagnosis_agent is None:
            from lettuce_dss.agents.diagnosis import DiagnosisAgent
            diagnosis_agent=DiagnosisAgent(self.root, mode=settings.get('model_mode','single'),
                                           device=settings.get('device','cpu'))
        if recommendation_agent is None:
            from lettuce_dss.agents.recommendation import RecommendationAgent
            recommendation_agent=RecommendationAgent(self.root)
        if quality_agent is None:
            from lettuce_dss.agents.image_quality import ImageQualityAgent
            quality_agent=ImageQualityAgent()
        self.agents={'environment':environment_agent,'diagnosis':diagnosis_agent,
                     'recommendation':recommendation_agent,'image_quality':quality_agent,
                     'reporting':reporting_agent or ReportingAgent()}

    def run(self, image_bytes:bytes, farm:dict) -> dict:
        if not isinstance(farm, dict):
            raise ValueError('Farm input must be an object.')
        farm = copy.deepcopy(farm)
        start=time.perf_counter();run_id=uuid.uuid4().hex[:16]
        results={};messages=[]
        def publish(key,result):
            results[key]=result
            messages.append({'kind':'inform','sender':key,'recipient':'shared_state',
                             'status':result.get('status'),'timestamp':utc_now(),
                             'payload':copy.deepcopy(result)})
        def call(key, *args, **kwargs):
            """A failed agent must not destroy other evidence or leak internal errors."""
            try:
                result = self.agents[key].run(*args, **kwargs)
                if (not isinstance(result, dict) or not isinstance(result.get('data'), dict)
                        or result.get('status') not in {'ok','partial','blocked','error'}):
                    raise ValueError('Invalid agent packet')
                return result
            except Exception:
                message = f'The {key.replace("_", " ")} agent could not complete this step.'
                data = {'uncertainty':True,'recommendation_scope':'general_only'}
                if key == 'image_quality':
                    data.update(usable=False, request_retake=False)
                if key == 'environment':
                    data.update(missing=['environmental context'], measurements={},
                        weather={'status':'unavailable'}, environment=farm.get('environment'),
                        root_system=farm.get('root_system'))
                if key in {'recommendation','reporting'}:
                    data.update(actions=[], all_actions=[], medication={'status':'not_selected','options':[]},
                                missing_context=[key + ' result'], feedback_request=None)
                if key == 'reporting':
                    data.update(headline='A report could not be prepared for this run.',
                                diagnosis_usable=False, generated_at=utc_now())
                return {'agent':key,'goal':getattr(self.agents[key],'goal','Return an explicit result'),
                        'status':'error','data':data,'elapsed_ms':0,'timestamp':utc_now(),
                        'decisions':[{'action':'withhold_failed_result','reason':message}],
                        'messages':[message]}
        with tempfile.TemporaryDirectory(prefix='lettuce_case_') as temp:
            with ThreadPoolExecutor(max_workers=1) as pool:
                context_future=pool.submit(call,'environment',farm)
                quality=call('image_quality',image_bytes,Path(temp)/'leaf.jpg')
                publish('image_quality',quality)
                if (quality.get('status') in {'blocked','error'}
                        or not quality.get('data',{}).get('usable',False)
                        or not isinstance(quality.get('data',{}).get('image_path'), str)):
                    publish('diagnosis',{'agent':'diagnosis','status':'blocked',
                        'data':{'uncertainty':True,'recommendation_scope':'general_only'},
                        'decisions':[{'action':'skip_prediction','reason':'Image quality did not permit prediction.'}],
                        'messages':[],'elapsed_ms':0,'timestamp':utc_now()})
                else:
                    prediction=call('diagnosis',Path(quality['data']['image_path']))
                    publish('diagnosis',prediction)
                try:
                    context=context_future.result()
                except Exception:
                    context={'agent':'environment','status':'error','data':{'missing':['environmental context'],
                            'measurements':{},'weather':{'status':'unavailable'},'environment':farm.get('environment'),
                            'root_system':farm.get('root_system')},'elapsed_ms':0,
                            'decisions':[{'action':'continue_without_context','reason':'Environmental agent could not complete.'}],
                            'messages':[],'timestamp':utc_now()}
                publish('environment',context)
        # Temporary image is deleted after inference; it is not persisted in logs.
        for packet in [results['image_quality'],messages[0]['payload']]:
            packet.get('data',{}).pop('image_path',None)
        rec=call('recommendation',results['diagnosis'],results['environment'],farm)
        publish('recommendation',rec)
        report=call('reporting',results,farm)
        request=report.get('data',{}).get('feedback_request')
        feedback_rounds=0
        if request and request.get('recipient') in {'recommendation','environment'}:
            messages.append(copy.deepcopy(request));feedback_rounds=1
            recipient=request['recipient'];payload=request.get('payload',{})
            if recipient=='environment':
                publish('environment',call('environment',farm,request=payload))
                publish('recommendation',call('recommendation',results['diagnosis'],results['environment'],farm))
            else:
                publish('recommendation',call('recommendation',results['diagnosis'],results['environment'],farm,request=payload))
            report=call('reporting',results,farm,feedback_used=True)
        publish('reporting',report)
        return {'run_id':run_id,'created_at':utc_now(),'app_version':__version__,
                'farm':copy.deepcopy(farm),'agents':results,'messages':messages,
                'feedback_rounds':feedback_rounds,
                'elapsed_seconds':round(time.perf_counter()-start,3),
                'image_sha256':hashlib.sha256(image_bytes).hexdigest(),'report':report}
