"""Run from the project directory: python -m streamlit run app.py"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
from pathlib import Path
import streamlit as st
from lettuce_dss.coordinator import Coordinator, input_fingerprint
from lettuce_dss.components.geolocation import location_button
from lettuce_dss.location import search_pakistan_places
from lettuce_dss.agents.reporting import report_markdown, report_html, action_sentence, medication_display, medication_lines, medication_use_note, medication_evidence_lines, evidence_sources, recommendation_usage_lines
from lettuce_dss.context_display import environment_view, environment_summary, context_label, markdown_text
from lettuce_dss.model_display import model_readiness, model_confidence_label
from lettuce_dss.knowledge import load_sources
from lettuce_dss.crop_profile import CROP_DEFAULTS, crop_value_label, crop_profile_rows

ROOT=Path(__file__).resolve().parent
APP_CONFIG=json.loads((ROOT/'config/app.json').read_text())
MODEL_CONFIG=json.loads((ROOT/'config/model.json').read_text())
st.set_page_config(page_title=APP_CONFIG['app_name'],page_icon='🌱',layout='wide')
st.markdown('''<style>.block-container{max-width:1150px;padding-top:2rem}h1,h2,h3{color:#245138}.stButton>button[kind="primary"]{background:#286642;border-color:#286642}div[data-testid="stMetric"]{background:#f2f6ee;border-radius:9px;padding:12px}small{color:#647261}</style>''',unsafe_allow_html=True)
st.title(APP_CONFIG['app_name'])
st.caption('A leaf photograph, your farm conditions, and clear next steps.')

@st.cache_resource(show_spinner=False)
def model_resource(mode:str,device:str,config_revision:str):
    from lettuce_dss.agents.diagnosis import DiagnosisAgent
    return DiagnosisAgent(ROOT,mode=mode,device=device)

def mark_crop_selection(field):
    origins=dict(st.session_state.get('crop_input_provenance',{}))
    origins[field]='user_selected'
    st.session_state['crop_input_provenance']=origins

if 'crop_input_provenance' not in st.session_state:
    st.session_state['crop_input_provenance']={field:'ui_default' for field in CROP_DEFAULTS}

def crop_selectbox(label,field,options):
    return st.selectbox(label,options,index=options.index(CROP_DEFAULTS[field]),
        format_func=crop_value_label,key='crop_'+field,on_change=mark_crop_selection,args=(field,))

with st.sidebar:
    st.header('About your crop')
    st.caption('Default crop settings. Adjust these to match your crop.')
    environment=crop_selectbox('Where is it grown?','environment',['unknown','outdoor','greenhouse','tunnel','indoor'])
    root_system=crop_selectbox('Root environment','root_system',['unknown','soil','substrate','hydroponic'])
    growth_stage=crop_selectbox('Growth stage','growth_stage',['unknown','seedling','vegetative','near_harvest'])
    irrigation=crop_selectbox('Watering method','irrigation',['unknown','drip','overhead','manual','recirculating'])
    with st.expander('Model settings'):
        available_modes=MODEL_CONFIG.get('available_modes',['single'])
        mode=st.selectbox('Prediction mode',available_modes,index=available_modes.index(APP_CONFIG.get('model_mode','single')),format_func=lambda x:f"Single model (seed {MODEL_CONFIG['single_seed']})")
        device=st.selectbox('Processing device',['auto','cpu'],index=['auto','cpu'].index(APP_CONFIG.get('device','auto')),format_func=lambda x:'Automatic (GPU if available)' if x=='auto' else 'CPU')
        readiness=model_readiness(ROOT,mode)
        st.caption('Image model: '+readiness['display_name'])
        if readiness['calibration_status']=='not_fitted':
            st.caption('Confidence scores are uncalibrated. Decision thresholds are provisional.')
        if not readiness['ready']:
            for note in readiness['messages']:st.info(note)
            st.caption('Restore the included model file if it is missing.')

if not readiness['ready']:
    st.info(readiness['display_name']+' is selected, but its checkpoint or configuration is not ready. Install the selected checkpoint before requesting an image diagnosis.')

left,right=st.columns([1.05,1],gap='large')
with left:
    st.subheader('1. Add a lettuce image')
    upload=st.file_uploader('One clear leaf photograph',type=['jpg','jpeg','png','webp','bmp'])
    image_bytes=upload.getvalue() if upload else None
    if image_bytes:
        if len(image_bytes)<=15*1024*1024:
            try:st.image(image_bytes,caption='Your uploaded photograph',width=380)
            except Exception:st.info('This file could not be previewed. Choose another image.')
        else:st.info('Choose an image smaller than 15 MB.')
    confirms_lettuce=st.checkbox('This is a photograph of lettuce.',value=False)
    st.subheader('2. Farm location')
    st.caption('Use the crop plot location. City centres and campuses may have no soil-map coverage; nearby map values are not substituted.')
    location={'latitude':None,'longitude':None,'label':'','source':'not supplied'}
    method=st.radio('Location method',['Enter coordinates','Use my device','Search a town','Skip location'],horizontal=True)
    if method=='Enter coordinates':
        a,b=st.columns(2)
        lat_text=a.text_input('Latitude',placeholder='e.g. 31.5204')
        lon_text=b.text_input('Longitude',placeholder='e.g. 74.3587')
        label=st.text_input('Farm / district name (optional)',max_chars=120)
        if lat_text.strip() and lon_text.strip():
            try:location={'latitude':float(lat_text),'longitude':float(lon_text),'label':label,'source':'farmer coordinates'}
            except ValueError:st.caption('Enter coordinates as numbers.')
    elif method=='Use my device':
        geo=location_button()
        if isinstance(geo,dict) and geo.get('status')=='ok':
            st.caption(f"Location found: {geo['latitude']:.5f}, {geo['longitude']:.5f} · reported accuracy ±{geo.get('accuracy_m',0):.0f} m")
            if st.checkbox('This is the farm location.',key=f"confirm_geo_{geo.get('timestamp')}"):
                location={'latitude':geo['latitude'],'longitude':geo['longitude'],'label':'Confirmed farm location','source':'browser permission','accuracy_m':geo.get('accuracy_m')}
    elif method=='Search a town':
        query=st.text_input('Pakistan town or city',max_chars=120)
        if st.button('Find places'):
            try:st.session_state['places']=search_pakistan_places(query)
            except Exception:
                st.session_state['places']=[];st.info('Place search is unavailable. You can enter coordinates instead.')
        places=st.session_state.get('places',[])
        if places:
            selected=st.selectbox('Choose a place',range(len(places)),format_func=lambda i:places[i]['label'])
            if st.checkbox('Use this town as the approximate farm location.'):
                location=dict(places[selected])
        st.caption('Town coordinates are approximate and may land on an urban map cell. Use the actual farm coordinates for soil background and weather.')
    use_weather=st.checkbox('Use available outdoor weather information',value=True)

with right:
    st.subheader('3. Add your observations')
    root_observation=st.selectbox('How does the root zone look?',['unknown','dry','moist','wet'],format_func=lambda x:x.title()) if root_system!='hydroponic' else 'unknown'
    notes=st.text_area('What have you noticed? (optional)',placeholder='For example: symptoms started after several wet days.',max_chars=1500,height=95)
    st.caption('Notes are kept in the downloadable agent record. Automatic decisions use the structured fields below.')
    measurements={}
    with st.expander('Add measurements you already have',expanded=False):
        st.caption('Leave a field blank when you do not have a reading. Farmer readings take priority over estimates.')
        pakistan_today=dt.datetime.now(dt.timezone(dt.timedelta(hours=5))).date()
        sample_date=st.date_input('When were these readings taken?',value=pakistan_today,max_value=pakistan_today)
        sample_time=st.time_input('Time of readings (Pakistan time, optional)',value=None)
        st.caption('Add the time for recent temperature and humidity readings. Leave it blank if only the date is known.')
        if st.checkbox('I do not know the measurement date'):
            sample_date=None
        fields=[('ph','Nutrient solution pH' if root_system=='hydroponic' else 'Soil / substrate pH','pH'),
                ('ec','Nutrient solution EC' if root_system=='hydroponic' else 'Soil / substrate EC','dS/m'),
                ('root_temperature','Root-zone temperature','C'),
                ('air_temperature','Air temperature at the crop','C'),
                ('relative_humidity','Relative humidity at the crop','%')]
        if root_system!='hydroponic':fields.insert(2,('moisture','Volumetric moisture','% VWC'))
        for key,label,unit in fields:
            text=st.text_input(f'{label} ({unit})',key='reading_'+root_system+'_'+key,placeholder='Optional reading')
            if text.strip():
                try:value=float(text)
                except ValueError:value=text
                measured_at=None
                if sample_date:
                    measured_at=dt.datetime.combine(sample_date,sample_time,dt.timezone(dt.timedelta(hours=5))).isoformat() if sample_time else sample_date.isoformat()
                measurements[key]={'value':value,'unit':unit,'measured_at':measured_at}

farm={'location':location,'environment':environment,'root_system':root_system,'growth_stage':growth_stage,
      'irrigation':irrigation,'input_provenance':dict(st.session_state['crop_input_provenance']),
      'measurements':measurements,'observations':{'root_zone':root_observation,'notes':notes},'use_weather':use_weather}
try:
    knowledge_sources=load_sources(ROOT)
    knowledge_revision=hashlib.sha256((ROOT/'knowledge/treatment_knowledge_base.json').read_bytes()).hexdigest()
    knowledge_ready=True
except (OSError,ValueError,TypeError,KeyError) as exc:
    knowledge_sources={};knowledge_revision='unavailable';knowledge_ready=False
    st.error('The treatment knowledge base is missing or invalid. Restore the supplied database before requesting recommendations.')
    with st.expander('Knowledge base error for troubleshooting'):st.code(f'{type(exc).__name__}: {exc}')
fingerprint=hashlib.sha256((input_fingerprint(image_bytes or b'',farm,mode,device)+readiness['revision']+knowledge_revision).encode()).hexdigest()
run=st.button('Get recommendations',type='primary',disabled=not(image_bytes and confirms_lettuce and readiness['ready'] and knowledge_ready))
if run:
    with st.spinner('Assessing the image and farm conditions…'):
        try:
            diagnosis=model_resource(mode,device,readiness['revision'])
            from lettuce_dss.agents.environment import EnvironmentalAgent
            environment_agent=EnvironmentalAgent(ROOT/'runtime/weather_cache',timeout=APP_CONFIG.get('weather_timeout_seconds',5.0))
            packet=Coordinator(ROOT,diagnosis_agent=diagnosis,environment_agent=environment_agent).run(image_bytes,farm)
            st.session_state['last_case']={'fingerprint':fingerprint,'packet':packet}
        except Exception as exc:
            st.error('The application could not finish this request. Check the installation steps in README.md.')
            with st.expander('Technical error for troubleshooting'):st.code(f'{type(exc).__name__}: {exc}')

case=st.session_state.get('last_case')
if case and case['fingerprint']!=fingerprint:
    st.info('The inputs have changed. Select Get recommendations for an updated report.')
elif case:
    packet=case['packet']; report=packet['report'];d=report['data'];sources=knowledge_sources
    st.divider();st.subheader('Your recommendations')
    if report['status']=='blocked':st.info(d['headline'])
    else:st.write(d['headline'])
    diag=packet['agents'].get('diagnosis',{}).get('data',{})
    a,b,c=st.columns(3)
    a.metric('Image result',str(d.get('prediction') or 'Unavailable').replace('_',' '))
    b.metric(model_confidence_label(diag),f"{d['confidence']:.1%}" if d.get('confidence') is not None else 'Unavailable')
    c.metric('Response time',f"{packet['elapsed_seconds']:.2f} s")
    if d.get('recommendation_notice'):st.info(d['recommendation_notice'])
    if diag.get('message'):st.caption(diag['message'])
    if diag.get('error'):st.info(str(diag['error']))
    for note in packet['agents'].get('diagnosis',{}).get('messages',[]):
        if isinstance(note,str):st.info(note)
    for note in packet['agents'].get('image_quality',{}).get('messages',[]):
        if isinstance(note,str):st.caption(note)
    with st.container(border=True):
        st.markdown('### Recommended actions')
        if d.get('actions'):
            for action in d['actions'][:4]:
                st.markdown('- '+markdown_text(action_sentence(action)))
        else:
            st.write('No actions selected.')
    with st.container(border=True):
        st.markdown('### Possible treatment options')
        medication=medication_display(d)
        med_lines,_=medication_lines(medication)
        st.write(medication['summary'])
        for line in med_lines:
            st.markdown('- '+markdown_text(line))
        if medication_use_note(medication):st.write(medication_use_note(medication))
    with st.container(border=True):
        st.markdown('### Soil/weather summary')
        for line in environment_summary(d.get('environment_summary',{}),d.get('farm_setup')):
            st.write(line)
    with st.expander('Evidence and details',expanded=False):
        env=packet['agents'].get('environment',{}).get('data',{})
        view=environment_view(env,d.get('farm_setup'),d.get('location'))
        st.text('Growing setup: '+view['setup']+'\nFarm location: '+view['location'])
        st.dataframe(crop_profile_rows(d.get('farm_setup',{})),hide_index=True,use_container_width=True)
        st.caption('Default assumptions describe the initial form settings, not confirmed observations.')
        st.caption('Times retain their recorded UTC offsets. Model estimates describe the outdoor model grid.')
        if view['measurements']:st.dataframe(view['measurements'],hide_index=True,use_container_width=True)
        st.text('Outdoor weather: '+'; '.join(view['weather_info']))
        if view['weather']:st.dataframe(view['weather'],hide_index=True,use_container_width=True)
        if view.get('soil_map'):
            st.text('Background soil map: '+'; '.join(view['soil_map_info']))
            st.caption('Geographic soil background; this does not describe a nutrient solution or imported substrate.')
            st.dataframe(view['soil_map'],hide_index=True,use_container_width=True)
        st.text(context_label('farmer_root_zone_observation')+': '+view['observation'])
        for note in view['notes']:st.text(note)
        if d.get('missing_context'):st.text('Unavailable: '+', '.join(context_label(k) for k in d['missing_context']))
        st.markdown('#### Action details')
        for action in d.get('all_actions',d.get('actions',[])):
            st.write(action.get('title','Action')+': '+action.get('detail',''))
            st.caption('Reason: '+action.get('reason',''))
            st.caption('Rules: '+', '.join(action.get('triggered_rule_ids',[]))+' · Sources: '+', '.join(action.get('source_ids',[])))
            for source in evidence_sources(action,sources):
                if source['url']:st.link_button(source['id']+': '+source['title'],source['url'])
        st.markdown('#### Treatment reference details')
        for option in medication.get('options',[]):
            st.write(option.get('active_ingredient','Reference')+': '+str(option.get('reason') or option.get('purpose','')))
            st.caption('Rules: '+', '.join(option.get('triggered_rule_ids',[]))+' · Sources: '+', '.join(option.get('source_ids',[])))
            for line in medication_evidence_lines(option):st.caption(line)
            for source in evidence_sources(option,sources):
                if source['url']:st.link_button(source['id']+': '+source['title'],source['url'])
        if d.get('rule_trace'):
            st.markdown('#### Selected, excluded and deferred options')
            st.dataframe(d['rule_trace'],hide_index=True,use_container_width=True)
        if d.get('knowledge_base'):
            kb=d['knowledge_base']
            st.caption('Knowledge base: '+str(kb.get('version',''))+' · SHA-256: '+str(kb.get('sha256','')))
            st.json(kb.get('counts',{}))
        for usage_line in recommendation_usage_lines(d):st.caption(usage_line)
        st.markdown('#### Sources')
        for sid in d.get('source_ids',[]):
            source=sources.get(sid,{})
            if isinstance(source,dict):
                url=str(source.get('url',''))
                if url.startswith(('https://','http://')):
                    st.markdown('- '+markdown_text(source.get('title',sid))+': '+markdown_text(url))
        st.caption('Weather data: Open-Meteo and its national weather-model providers (CC BY 4.0).')
        st.markdown('#### Agent activity')
        st.dataframe([{'Agent':key.replace('_',' ').title(),'Status':r.get('status'),'Milliseconds':r.get('elapsed_ms'),
                       'Decisions':'; '.join(a.get('action','') for a in r.get('decisions',[]))} for key,r in packet['agents'].items()],hide_index=True,use_container_width=True)
        st.caption(f"One bounded feedback round at most. This request used {packet['feedback_rounds']}.")
        model=d.get('diagnosis_model',{})
        st.caption('Image model: '+str(model.get('display_name') or model.get('model_id') or readiness['display_name'])+
                   ' · Mode: '+str(model.get('model_mode','unavailable'))+
                   ' · Calibration: '+str(model.get('calibration_status','unavailable')).replace('_',' '))
        st.json({'model':diag,'messages':packet['messages']})
    a,b,c=st.columns(3)
    a.download_button('Download readable report',report_html(report,sources),file_name='lettuce_recommendations.html',mime='text/html')
    b.download_button('Download report text',report_markdown(report,sources),file_name='lettuce_recommendations.md',mime='text/markdown')
    c.download_button('Download agent record',json.dumps(packet,indent=2,default=str),file_name='lettuce_agent_record.json',mime='application/json')
st.divider()
st.caption('Lettuce decision support · International treatment references · Farmer measurements first')
