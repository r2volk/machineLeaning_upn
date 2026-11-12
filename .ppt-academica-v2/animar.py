"""Añade transiciones y entradas suaves nativas a la exportación Artifact Tool.
No altera el contenido, las notas ni la geometría de los objetos.
Esquema: Microsoft PresentationML transition/timing/animEffect.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import re,json
from xml.etree import ElementTree as ET
BASE=Path(__file__).resolve().parent
P='http://schemas.openxmlformats.org/presentationml/2006/main'
ns={'p':P}
manifest=[]
with ZipFile(BASE/'draft.pptx') as source, ZipFile(BASE/'candidate.pptx','w',ZIP_DEFLATED) as dest:
    for item in source.infolist():
        data=source.read(item.filename)
        if re.fullmatch(r'ppt/slides/slide\d+\.xml',item.filename):
            xml=data.decode('utf-8');root=ET.fromstring(xml)
            targets=[el.get('id') for el in root.findall('.//p:cNvPr',ns)
                     if el.get('name','').startswith('reveal-')]
            transition='<p:transition spd="med" advClick="1"><p:fade/></p:transition>'
            effects=[];index=4
            for n,spid in enumerate(targets):
                delay=80+n*45
                effects.append(f'''<p:par><p:cTn id="{index}" presetID="10" presetClass="entr" presetSubtype="0" fill="hold" nodeType="withEffect"><p:stCondLst><p:cond delay="{delay}"/></p:stCondLst><p:childTnLst><p:set><p:cBhvr><p:cTn id="{index+1}" dur="1" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl><p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst></p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set><p:animEffect transition="in" filter="fade"><p:cBhvr><p:cTn id="{index+2}" dur="450" fill="hold"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr></p:animEffect></p:childTnLst></p:cTn></p:par>''')
                index+=3
            timing=''
            if effects:
                timing='''<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst><p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst><p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>'''+''.join(effects)+'''</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn><p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst></p:seq></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>'''
            xml=xml.replace('</p:sld>',transition+timing+'</p:sld>')
            final=ET.fromstring(xml)
            ids=[el.get('id') for el in final.findall('.//p:cTn',ns)]
            assert len(ids)==len(set(ids))
            assert final.find('p:transition',ns) is not None
            data=xml.encode('utf-8')
            manifest.append({'slide':item.filename,'transition':'fade','entrances':len(targets)})
        dest.writestr(item,data)
(BASE/'animaciones.json').write_text(json.dumps(manifest,indent=2))
print('Transiciones:',len(manifest),'Entradas suaves:',sum(x['entrances'] for x in manifest))
