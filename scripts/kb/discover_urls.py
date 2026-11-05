import sys,re,json,time,requests,random,glob
from urllib.parse import urljoin
from bs4 import BeautifulSoup
sys.stdout.reconfigure(encoding='utf-8',line_buffering=True)
H={"User-Agent":"Mozilla/5.0 (compatible; TruthCheckResearchBot/0.1; academic health misinformation project)"}
def get(u):
    time.sleep(random.uniform(1.5,2.5)); return requests.get(u,headers=H,timeout=40)
def locs(u):
    try:
        r=get(u); return re.findall(r'<loc>\s*(.*?)\s*</loc>',r.text)
    except Exception as e: print('ERR',u,str(e)[:100]); return []
out={}
idx=locs("https://www.cdc.gov/wcms-auto-sitemap-index.xml")
want=re.compile(r'cdc\.gov/(covid|vaccines|vaccine-safety|vaccines-children|vaccines-adults|vaccines-pregnancy|vaccine-for-children|flu|bird-flu|ebola|marburg|measles|mpox|polio|hpv|whooping-cough|rsv|nipah|zika|dengue|malaria|tuberculosis|hepatitis|rabies|cholera|antibiotic-use)/wcms-auto-sitemap')
cdc=[]
for c in idx:
    if want.search(c):
        l=locs(c); print('cdc',c.split('/')[3],len(l)); cdc+=l
out['cdc']=cdc
n=locs("https://www.nhs.uk/sitemap-cms-content.xml"); print('nhs all',len(n)); out['nhs']=n
m=locs("https://medlineplus.gov/sitemap.xml"); out['medline']=[u for u in m if '/lab-tests/' in u]; print('medline tests',len(out['medline']))
who=set()
for p in ["https://www.who.int/news-room/questions-and-answers","https://www.who.int/news-room/fact-sheets","https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters","https://www.who.int/emergencies/diseases/novel-coronavirus-2019/question-and-answers-hub"]:
    r=get(p); print('who',p.split('/')[-1],r.status_code,len(r.text))
    for a in BeautifulSoup(r.text,'html.parser').find_all('a',href=True):
        u=urljoin(p,a['href']).split('#')[0]
        if re.search(r'who\.int/(news-room/(questions-and-answers/item|fact-sheets/detail)|emergencies/diseases/novel-coronavirus-2019/(question-and-answers-hub/q-a-detail|advice-for-public))',u): who.add(u)
out['who']=sorted(who); print('who urls',len(who))
vi=[]
for s in locs("https://www.vishvasnews.com/sitemap_index.xml"):
    if 'post-sitemap' in s:
        l=[u for u in locs(s) if re.search(r'/english/health/|/health/',u)]; print('vishvas',s.split('/')[-1],len(l)); vi+=l
out['vishvas']=vi
for name,u in {"icmr":"https://www.icmr.gov.in/sitemap.xml","fssai":"https://fssai.gov.in/sitemap.xml"}.items():
    l=locs(u); print(name,len(l),l[:15]); out[name]=l
json.dump(out,open(sys.argv[1],'w'),indent=0); print('DONE',{k:len(v) for k,v in out.items()})
