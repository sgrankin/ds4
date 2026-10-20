import hashlib,json,os,pathlib,stat,subprocess,time
ROOT=pathlib.Path('/Volumes/Models/ds4/.workspaces/idea-practical-write-observation-v4')
PKG=ROOT/'speed-bench/practical-write-native-controls-v4'
FREEZE='0f6dd6e8d0bcbe3dd4418bc738fcf9fbf5175263'
MANIFEST='f51320fe439cc2a4be23dbe9bdae602882d0a05ef748efbf7a0dbc7bf138515d'
SOURCEPLAN='ddb7e9a31952a26bc5c8637a3c517b5e5049cd6d5e66c562db618c09b8bbff0f'
CPUPLAN='b8672b7dbf73a7c0266143338f2e29763817fd5d4d271103eafe3c9255843475'
OUT=ROOT/'speed-bench/ideas/artifacts/write-native-controls-source-cpu-v4'
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sha(PKG/'manifest.json')==MANIFEST and sha(PKG/'source-plan.json')==SOURCEPLAN and sha(PKG/'CPU-plan.json')==CPUPLAN
manifest=json.loads((PKG/'manifest.json').read_text());source=json.loads((PKG/'source-plan.json').read_text());plan=json.loads((PKG/'CPU-plan.json').read_text())
literal={PKG/name:pin for name,pin in manifest['files_excluding_this_manifest'].items()};literal[PKG/'manifest.json']={'sha256':MANIFEST,'bytes':(PKG/'manifest.json').stat().st_size,'mode':420}
assert len(literal)==25
for p,pin in literal.items():
 assert stat.S_ISREG(p.lstat().st_mode) and (p.stat().st_mode&0o777)==pin['mode'] and p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256']
 shown=subprocess.run(['jj','--ignore-working-copy','--color','never','file','show','-r',FREEZE,str(p.relative_to(ROOT))],cwd=ROOT,capture_output=True,check=True).stdout
 assert shown==p.read_bytes()
tree=subprocess.run(['jj','--ignore-working-copy','--color','never','debug','tree','-r',FREEZE,*['file:'+str(p.relative_to(ROOT)) for p in literal]],cwd=ROOT,capture_output=True,check=True,text=True).stdout
for p in literal:
 lines=[x for x in tree.splitlines() if x.startswith(str(p.relative_to(ROOT))+': ')]
 assert len(lines)==1 and 'Resolved(Some(File {' in lines[0] and 'executable: false' in lines[0]
assets=source['assets'];assert len(assets)==208
LAUNCHER=pathlib.Path('/opt/homebrew/opt/python@3.14/bin/python3.14')
LAUNCHER_TARGET=pathlib.Path('/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/bin/python3.14')
LIB=pathlib.Path('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX27.0.sdk/usr/lib/libSystem.tbd')
LIB_TARGET=pathlib.Path('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/usr/lib/libSystem.B.tbd')
def verify():
 for name,pin in assets.items():
  p=pathlib.Path(name);assert p.suffix.lower()!='.gguf'
  if p==LAUNCHER:
   assert p.is_symlink() and os.readlink(p)=='../Frameworks/Python.framework/Versions/3.14/bin/python3.14' and p.resolve()==LAUNCHER_TARGET
  elif p==LIB:
   assert p.is_symlink() and os.readlink(p)=='libSystem.B.tbd' and p.resolve()==LIB_TARGET and not LIB_TARGET.is_symlink()
  else:assert stat.S_ISREG(p.lstat().st_mode),name
  s=p.stat();assert s.st_size==pin['bytes'] and stat.S_IMODE(s.st_mode)==pin['mode'] and sha(p)==pin['sha256'],name
 for p,pin in literal.items():assert sha(p)==pin['sha256']
verify();assert not OUT.exists() and len(plan['commands'])==1
cmd=plan['commands'][0];assert cmd['argv'][0]=='/opt/homebrew/Cellar/python@3.14/3.14.8/Frameworks/Python.framework/Versions/3.14/Resources/Python.app/Contents/MacOS/Python'
assert cmd['argv']==[cmd['argv'][0],'-B',str(PKG/'test_controls.py')] and cmd['cwd']==str(PKG) and cmd['timeout_seconds']==60
for path in plan['outputs_must_not_exist']:assert not pathlib.Path(path).exists()
OUT.mkdir();(OUT/'temporary').mkdir()
grant=dict(status='ROOT_GRANTS_ONE_EXACT_CPU17_COMMAND',source_freeze=FREEZE,manifest_sha256=MANIFEST,source_plan_sha256=SOURCEPLAN,CPU_plan_sha256=CPUPLAN,command=cmd,literal_source_files=len(literal),asset_pins=len(assets),no_model_helper_compiler_or_GPU=True,time=time.time())
(OUT/'root-grant.json').write_text(json.dumps(grant,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))};env.update(cmd['env']);env['TMPDIR']=str(OUT/'temporary');start=time.monotonic()
with (OUT/'stdout').open('xb') as stdout,(OUT/'stderr').open('xb') as stderr:
 try:process=subprocess.run(cmd['argv'],cwd=cmd['cwd'],env=env,stdout=stdout,stderr=stderr,timeout=60);rc=process.returncode
 except subprocess.TimeoutExpired:rc=124
seconds=time.monotonic()-start;verify();stderr=(OUT/'stderr').read_text()
passed=rc==0 and 'Ran 17 tests' in stderr and stderr.rstrip().endswith('OK')
proof=dict(status='WRITE_NATIVE_CONTROL_CPU_PASS' if passed else 'FIRST_FAILURE_RETAINED',returncode=rc,seconds_diagnostic=seconds,tests=17 if passed else None,command=cmd,source_freeze=FREEZE,CPU_plan_sha256=CPUPLAN,source_plan_sha256=SOURCEPLAN,manifest_sha256=MANIFEST,grant_sha256=sha(OUT/'root-grant.json'),all_inputs_unchanged=True,stdout_sha256=sha(OUT/'stdout'),stderr_sha256=sha(OUT/'stderr'),native_parser_renderer_session_or_model_GPU_executed=False)
(OUT/'root-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({k:proof[k] for k in ('status','returncode','tests','seconds_diagnostic')})+' '+sha(OUT/'root-proof.json'))
raise SystemExit(0 if passed else 3)
