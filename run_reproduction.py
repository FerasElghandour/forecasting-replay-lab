"""Execute a versioned reproduction profile and retain exact commands and logs.

Profiles distinguish contract checks from training and full-data benchmarks. No
shell interpolation or implicit dataset/provider download occurs. Inputs must be
supplied explicitly; missing inputs never fall back to generated performance data.
"""
import argparse,hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--profile',default='checks');p.add_argument('--output',type=Path,required=True);p.add_argument('--input',action='append',default=[],metavar='NAME=PATH');a=p.parse_args();root=Path(__file__).resolve().parent;spec=json.loads((root/'reproduction.json').read_text())
 if a.output.exists():raise ValueError('Use a new output directory')
 if a.profile not in spec['profiles']:raise ValueError('Unknown profile; choose '+', '.join(spec['profiles']))
 given={}
 for item in a.input:
  name,value=item.split('=',1);path=Path(value).resolve()
  if not path.exists():raise ValueError('Missing input '+name)
  given[name]=path
 profile=spec['profiles'][a.profile]
 for name in profile.get('required_inputs',[]):
  if name not in given:raise ValueError('Profile requires --input '+name+'=PATH')
 out=a.output.resolve();out.mkdir(parents=True);results=[];started=time.time()
 def expand(value):
  value=value.replace('{python}',sys.executable).replace('{output}',str(out))
  for name,path in given.items():value=value.replace('{input:'+name+'}',str(path))
  if '{input:' in value:raise ValueError('Unresolved input in command')
  return value
 env={**os.environ,'PYTHONPATH':os.pathsep.join([str(root),str(root/'src'),os.environ.get('PYTHONPATH','')]),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','LOKY_MAX_CPU_COUNT':'2'}
 failed=False
 for i,command in enumerate(profile['commands']):
  cmd=[expand(x) for x in command];stamp=time.perf_counter()
  process=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True)
  (out/f'command-{i:02}.stdout.txt').write_text(process.stdout);(out/f'command-{i:02}.stderr.txt').write_text(process.stderr)
  results.append({'argv':cmd,'returncode':process.returncode,'seconds':time.perf_counter()-stamp})
  if process.returncode:failed=True;break
 versions={}
 for name in spec.get('version_packages',[]):
  try:versions[name]=importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:versions[name]=None
 tracked=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.py','.ipynb','.sql') and not any(c in p.parts for c in ('.venv','__pycache__','.git'))]
 record={'repository':spec['repository'],'profile':a.profile,'scope':profile['scope'],'status':'failed' if failed else 'passed','started_unix':started,'finished_unix':time.time(),'python':sys.version.split()[0],'versions':versions,'input_sha256':{n:hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else 'directory; individual data manifests in profile output' for n,p in given.items()},'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked},'commands':results}
 (out/'reproduction-record.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'repository':spec['repository'],'profile':a.profile,'status':record['status']}))
 if failed:raise SystemExit(1)
if __name__=='__main__':main()
