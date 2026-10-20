"""ROOT static preparation of uniformly observed WRITE controls; no launch."""
import hashlib,json,os,pathlib,stat,subprocess,time
MAIN=pathlib.Path('/Volumes/Models/ds4')
executor=MAIN/'speed-bench/ideas/root-write-control-cpu-v4.py'
text=executor.read_text();boundary='verify();assert not OUT.exists()';assert text.count(boundary)==1
exec(compile(text.split(boundary)[0],str(executor),'exec'))
verify()
cpu=ROOT/'speed-bench/ideas/artifacts/write-native-controls-source-cpu-v4/root-proof.json';assert sha(cpu)=='686139e69e506dccf810041ce0871c1ac6620ef746f4709a75363710eb6a3e34';cp=json.loads(cpu.read_bytes());assert cp['status']=='WRITE_NATIVE_CONTROL_CPU_PASS' and cp['tests']==17 and cp['returncode']==0 and cp['all_inputs_unchanged'] is True
old=MAIN/'.workspaces/idea-practical-write-native-control-runner-v1'
recipepath=old/'speed-bench/ideas/evidence/write-native-controls-CPU14-build2-20261003-v3/materialization-recipe.json';assert sha(recipepath)=='dd28397683f4113fb139d3a7e2f82832d671bfa1c5347971d3f779204d6350b1';recipe=json.loads(recipepath.read_bytes())
helper=recipe['teacher_helper'];bp=recipe['teacher_build_proof'];assert helper['sha256']=='6e173e56a90e84ec784301d411a4d75a4bee473551407c7a3b43966c9a5edfb5' and bp['sha256']=='70ace27e70ec3f20f02e89b7d43d846b7aad018e0b23af828d3831a92a888c5b'
proof=json.loads(pathlib.Path(bp['path']).read_bytes());build=json.loads((PKG/'native-build-plan.json').read_bytes());assert proof['status']=='WRITE_TEACHER_CALLER_BUILD_PASS' and proof['all_inputs_unchanged'] is True and proof['core_Metal_rebuilt'] is False and proof['plan_sha256']==sha(PKG/'native-build-plan.json') and len(proof['results'])==2
for row,c in zip(proof['results'],build['commands']):assert row['argv']==c['argv'] and row['cwd']==c['cwd'] and type(row['returncode']) is int and row['returncode']==0 and row['stderr_sha256']=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
def bytepin(p):
 p=pathlib.Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode)
 return dict(bytes=s.st_size,mode=stat.S_IMODE(s.st_mode),sha256=sha(p))
for r in (helper,bp):assert bytepin(r['path'])=={k:r[k] for k in ('bytes','mode','sha256')}
built=json.loads((PKG/'source-plan.json').read_bytes());built['status']='BUILT_PINNED';built['teacher_helper']=helper;built['teacher_build_proof']=bp;built['assets'].update({helper['path']:{k:helper[k] for k in ('bytes','mode','sha256')},bp['path']:{k:bp[k] for k in ('bytes','mode','sha256')},str(cpu):bytepin(cpu)});assert len(built['assets'])==211
assets=built['assets'];verify()
authoritypath=MAIN/'speed-bench/ideas/evidence/reasoning-budget-pilot-root-import-20261003.json';assert sha(authoritypath)=='2f4c1ee85cd8bf6c411366f7907c406de12d7e18aa2a4af2efc28de462538360';runtime=json.loads(authoritypath.read_bytes())['runtime_and_shaders_unchanged'];assert len(runtime)==100
def runtime_verify():
 for p,d in runtime.items():assert sha(MAIN/p)==d
runtime_verify();out=ROOT/'speed-bench/ideas/artifacts/practical-write-native-controls-prepared-v4';assert not out.exists();out.mkdir()
with (out/'plan.json').open('x') as f:json.dump(built,f,indent=2);f.write('\n')
digest=sha(out/'plan.json');command=[digest if x=='<PLAN_SHA256>' else x for x in built['guard_command_template']]
with (out/'prepared-command.json').open('x') as f:json.dump(dict(command=command,cwd=str(MAIN),plan_sha256=digest,no_autolaunch=True),f,indent=2);f.write('\n')
argv=[command[0],'-B',str(PKG/'run.py'),'--verify','--plan',str(out/'plan.json'),'--plan-sha256',digest]
grant=dict(status='ROOT_GRANTS_STATIC_VERIFIER_ONLY',time=time.time(),command=argv,cwd=str(ROOT),source_freeze=FREEZE,plan_sha256=digest,CPU17_proof_sha256=sha(cpu),reused_exact_teacher_build_proof_sha256=bp['sha256'],uniform_timing_summary_observation=True,ROOT_executor_sha256=sha(pathlib.Path(__file__)),native_helper_model_compiler_GPU_guard=False,no_retry=True)
with (out/'root-static-grant.json').open('x') as f:json.dump(grant,f,indent=2);f.write('\n')
started=time.monotonic()
with (out/'static-stdout').open('xb') as so,(out/'static-stderr').open('xb') as se:r=subprocess.run(argv,cwd=ROOT,env={**{k:v for k,v in os.environ.items() if not k.startswith(('DS4','PT_'))},'PYTHONDONTWRITEBYTECODE':'1'},stdout=so,stderr=se,timeout=60)
verify();runtime_verify();passed=r.returncode==0 and (out/'static-stdout').read_bytes()==b'STATIC PASS (no GGUF metadata/helper/guard)\n' and not (out/'static-stderr').stat().st_size
result=dict(status='WRITE_BUILT_PLAN_STATIC_PASS' if passed else 'FIRST_STATIC_FAILURE_RETAINED',returncode=r.returncode,wall_seconds_diagnostic=time.monotonic()-started,command=argv,plan_sha256=digest,source_freeze=FREEZE,original208_assets_unchanged=True,added_assets=3,total_assets=211,MAIN_runtime100_unchanged=True,root_static_grant_sha256=sha(out/'root-static-grant.json'),stdout_sha256=sha(out/'static-stdout'),stderr_sha256=sha(out/'static-stderr'),model_helper_guard_compiler_or_GPU_executed=False,model_grant_created=False,no_retry=True)
with (out/'root-static-proof.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ('status','returncode','wall_seconds_diagnostic','total_assets')}));print('PLAN',digest);print('COMMAND',sha(out/'prepared-command.json'));print('PROOF',sha(out/'root-static-proof.json'));raise SystemExit(0 if passed else 3)
