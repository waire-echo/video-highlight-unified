"""Versioned evidence contracts for the isolated temporal test."""
from pathlib import Path
PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts/ora/temporal-editor-test-v2"
for name in ("DISCOVER", "GLOBAL", "CORE", "BOUNDARY", "VERIFY", "DUPLICATE", "OBSERVE"):
    globals()[name] = (PROMPT_DIR / (name.lower() + ".txt")).read_text(encoding="utf-8").strip()

def obj(properties):
    return {'type':'object','properties':properties,'required':list(properties),'additionalProperties':False}

def text(limit=240): return {'type':'string','minLength':1,'maxLength':limit,
    'description':'Describe the requested visible fact, uncertainty or reason in words. Not a timestamp, numeric score or numeric placeholder.'}
def enum(*values): return {'type':'string','enum':list(values)}
NUMBER={'type':'number'}
NULLABLE={'type':['number','null']}
TIMES={'type':'array','items':NUMBER,'minItems':1,'maxItems':4}
DISCOVERY_SCHEMA=obj({'candidates':{'type':'array','maxItems':6,'items':obj({
    'start':NUMBER,'end':NUMBER,'event':text(160),'evidence':text(),
    'distinct_change':text(),'uncertainty':text()})},'overflow':{'type':'boolean'},'empty_reason':text()})
CORE_SCHEMA=obj({'decision':enum('keep','drop','review'),'start':NULLABLE,'end':NULLABLE,
    'observed_event':text(),'hypothesis_match':enum('supported','contradicted','uncertain'),
    'evidence_before':text(),'evidence_core':text(),'evidence_after':text(),
    'reason':text(),'evidence_times':TIMES})
BOUNDARY_SCHEMA=obj({'decision':enum('adjust','unchanged','uncertain'),'time':NULLABLE,
    'reason':text(),'evidence_time':NUMBER})
VERIFY_SCHEMA=obj({'decision':enum('keep','drop','review'),'reason':text(),'evidence_times':TIMES})
DUPLICATE_SCHEMA=obj({'decision':enum('duplicate','distinct','uncertain'),
    'representative':enum('first','second','both','uncertain'),'reason':text()})
OBSERVE_SCHEMA=obj({'subject':text(),'observed_event':text(),'uncertainty':text(),
    'evidence_times':TIMES})
