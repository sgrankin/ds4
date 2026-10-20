"""ROOT actual-artifact CPU publication, followed by independent byte audit.
Never opens/stats model weights or invokes a helper/GPU/tokenizer.
"""
import base64,hashlib,json,math,os,pathlib,stat,struct,sys,time,traceback
MAIN=pathlib.Path('/Volumes/Models/ds4');WS=MAIN/'.workspaces/idea-resident-functional-map-publication'
B=WS/'speed-bench/ideas/resident_functional_map_publication_cpu_prepared_v1'
OUT=WS/'speed-bench/ideas/artifacts/resident-functional-map-publication-v1'
SCAN=MAIN/'.workspaces/idea-resident-functional-constructor-run/speed-bench/ideas/artifacts/resident-functional-fullscan-v3'
PLAN=MAIN/'.workspaces/idea-resident-functional-constructor-run/speed-bench/ideas/resident_functional_fullscan_prepared_v3/plan.json'
CAP=MAIN/'.workspaces/idea-resident-functional-capture-offline/speed-bench/ideas/artifacts/resident-functional-capture-offline-publication-v2/capture.ds4fcap1'
AUTH_SHA='0f4c3469798a3d8e9677b7111c92429b717ce9a2d378ff5ae6bde58746ece1db'
def h(raw):return hashlib.sha256(raw).hexdigest()
def sha(p):
    p=pathlib.Path(p);assert p.suffix.lower()!='.gguf'
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):
    p=pathlib.Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode)
    return dict(bytes=s.st_size,mode=stat.S_IMODE(s.st_mode),sha256=sha(p))
def put(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def canon(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')
def pairs(xs):
    d={}
    for k,v in xs:assert k not in d;d[k]=v
    return d
def decode(raw):return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite')))
def identity(s):return (s.st_dev,s.st_ino,s.st_mode,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def owned_read(p,r):
    p=pathlib.Path(p);assert p.suffix.lower()!='.gguf';before=p.lstat();assert stat.S_ISREG(before.st_mode) and before.st_size==r['bytes'] and stat.S_IMODE(before.st_mode)==r['mode']
    fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        assert identity(os.fstat(fd))==identity(before)
        with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(r['bytes']+1)
        assert len(raw)==r['bytes'] and h(raw)==r['sha256'] and identity(os.fstat(fd))==identity(before) and identity(p.lstat())==identity(before)
    finally:os.close(fd)
    return raw
cpu=B.parent/'artifacts/resident-functional-map-publication-cpu-v1/root-proof.json'
assert sha(cpu)=='4528b2a2969252ed8b5076ea840849597cc5a06f0d9444143fa66b0396c7df43' and decode(cpu.read_bytes())['status']=='FUNCTIONAL_MAP_PUBLICATION_SYNTHETIC_CPU16_PASS'
idx=decode((B/'physical-index.json').read_bytes());sourcepins={B/n:r for n,r in idx['indexed_files'].items()};sourcepins[B/'physical-index.json']=pin(B/'physical-index.json')
for p,r in sourcepins.items():assert pin(p)==r
assert sha(B/'source-authority.json')==AUTH_SHA
sys.path.insert(0,str(B));import resident_functional_map_publish as publisher
source=(B/'source-authority.json').read_bytes();s=decode(source);planraw=owned_read(PLAN,dict(bytes=PLAN.lstat().st_size,mode=0o644,sha256=publisher.ACCEPTED['plan']));plan=decode(planraw)
proofpaths=dict(plan=PLAN,execution=SCAN/'root-execution-proof.json',release=SCAN/'root-release-proof.json',audit=SCAN/'root-independent-audit-proof.json',stdout=SCAN/'root-run.stdout',build=pathlib.Path(plan['native_build_proof']['path']))
proofpins={k:dict(path=str(p),bytes=p.lstat().st_size,mode=stat.S_IMODE(p.lstat().st_mode),sha256=publisher.ACCEPTED[k]) for k,p in proofpaths.items()}
for k,r in proofpins.items():assert pin(pathlib.Path(r['path']))=={a:r[a] for a in ('bytes','mode','sha256')}
assert not OUT.exists();OUT.mkdir(parents=True)
put(OUT/'root-grant.json',dict(status='ROOT_GRANTS_ACTUAL_ACCEPTED_ARTIFACT_MAP_PUBLICATION_CPU_ONLY',time=time.time(),source_freeze='22ca80e1bbeec688d5474069b1d39216d79de6b5',source_authority_sha256=AUTH_SHA,CPU16_proof_sha256=sha(cpu),capture=dict(path=str(CAP),bytes=13290484,mode=0o444,sha256=publisher.CAPTURE['sha256']),signature=dict(path=str(SCAN/'signatures.fp16le'),bytes=7864320,mode=0o644,sha256=publisher.ACCEPTED['signature']),proofs=proofpins,ROOT_executor_sha256=sha(__file__),no_current_model_FD_metadata_helper_guard_GPU_tokenizer=True,no_retry=True))
start=time.monotonic();passed=False
try:
    capture=owned_read(CAP,dict(bytes=13290484,mode=0o444,sha256=publisher.CAPTURE['sha256']));sig=owned_read(SCAN/'signatures.fp16le',dict(bytes=7864320,mode=0o644,sha256=publisher.ACCEPTED['signature']))
    proofs={k:owned_read(pathlib.Path(r['path']),r) for k,r in proofpins.items()}
    mapped,companion,authority=publisher.compose(capture,sig,proofs,source,source_authority_sha256=AUTH_SHA)
    for name,raw in [('functional.ds4fmap1',mapped),('qualification.ds4fqal1',companion)]:
        p=OUT/name
        with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        p.chmod(0o444)
    # Independent fixed-format reader: no publisher/reference/qualification parser.
    mr=owned_read(OUT/'functional.ds4fmap1',dict(bytes=len(mapped),mode=0o444,sha256=h(mapped)));qr=owned_read(OUT/'qualification.ds4fqal1',dict(bytes=len(companion),mode=0o444,sha256=h(companion)))
    fields=struct.unpack_from('<8s10I5Q32s32s32s32s40s',mr)
    assert fields[:11]==(b'DS4FMAP1',1,256,40,384,16,5120,16,32,16,1) and fields[11]==0x445334464d415031 and fields[-1]==bytes(40)
    mb,pb,sb,total=fields[12:16];assert 0<mb<=1<<20 and pb==13107200 and sb==7864320 and total==mb+pb+sb and len(mr)==256+total
    rawmeta=mr[256:256+mb];probes=mr[256+mb:256+mb+pb];signatures=mr[256+mb+pb:]
    assert tuple(hashlib.sha256(x).digest() for x in (rawmeta,probes,signatures))==fields[16:19] and signatures==sig
    meta=decode(rawmeta);assert canon(meta)==rawmeta
    cap_meta_n=struct.unpack_from('<Q',capture,48)[0];capmeta=decode(capture[256:256+cap_meta_n]);cp=256+cap_meta_n;assert probes==capture[cp:cp+13107200]
    hist=capture[cp+13107200:];histories=[]
    for i,x in enumerate(capmeta['probes']):
        at=x['history_byte_offset'];opaque=hist[at:at+x['history_count']*4]
        assert h(opaque)==x['history_sha256'] and len(opaque)==x['position']*4 and struct.unpack_from('<I',opaque,len(opaque)-4)[0]==x['evaluated_token']
        histories.append(dict(probe=i,case=x['case_id'],history_le32_base64=base64.b64encode(opaque).decode('ascii'),history_sha256=x['history_sha256'],position=x['position'],evaluated_token=x['evaluated_token']))
    n=capmeta['native_binding'];native=struct.pack('<8sIIQQQQIIIIII',b'DS4FMBN1',1,120,n['whole_file_bytes'],n['addressable_bytes'],n['tensor_data_pos'],n['effective_cache_bytes'],n['context'],n['prefill_chunk'],n['configured_cache_slots'],n['fused_norm'],n['clamp_bits'],n['rms_eps_bits'])
    tensors=[]
    for layer,roles in enumerate(n['tensor']):
        row={'layer':layer}
        for role,t in zip(('gate','up','down'),roles):
            native+=struct.pack('<QQQQIIII',t['abs_offset'],t['row_bytes'],t['expert_bytes'],t['tensor_bytes'],t['type'],t['columns'],t['rows'],t['experts'])
            row[role]=dict(offset=t['abs_offset'],bytes=t['tensor_bytes'],type=t['type'],row_bytes=t['row_bytes'],expert_bytes=t['expert_bytes'])
        tensors.append(row)
    assert len(native)==5832 and h(native)==capmeta['native_binding_le_sha256']
    expected_binding=dict(model=capmeta['identity']['original_model_literal_no_new_stat'],shape=[40,384,16,5120,16],tensors=tensors,provenance=s['map_provenance'],projection=dict(seed_hex='445334464d415031',algorithm='splitmix64_v1',normalization='1/sqrt(5120)',accumulation='FP64_ascending_left_to_right',rounding='FP16_RNE'),case_histories=histories,norm_kernel='fused_RMS_BF16' if n['fused_norm'] else 'default_RMS_BF16')
    assert meta['binding']==expected_binding and hashlib.sha256(canon(expected_binding)).digest()==fields[19]
    expected_captures=[dict(layer=layer,probe=i,case=x['case'],position=x['position'],evaluated_token=x['evaluated_token'],history_sha256=x['history_sha256'],teacher_evaluation_index=0,input_after_existing_router_completion=True) for layer in range(40) for i,x in enumerate(histories)]
    assert meta['captures']==expected_captures and meta['schema']=='DS4_FUNCTIONAL_MATCH_V1'
    assert meta['construction']==dict(requested_bytes=152882380800,returned_bytes=152882380800,owned_peak_bytes=31735808,seconds=decode(proofs['execution'])['wall_seconds_diagnostic'])
    assert all(bits&0x7f800000!=0x7f800000 for bits, in struct.iter_unpack('<I',probes)) and all(bits&0x7c00!=0x7c00 for bits, in struct.iter_unpack('<H',signatures))
    assert qr[:8]==b'DS4FQAL1' and struct.unpack_from('<8I',qr,8)==(1,512,5832,8192,512,64,1024,0) and qr[296:512]==bytes(216)
    mapbytes,dev,ino,whole,mtime,ctime,qn,reserved=struct.unpack_from('<8Q',qr,40);assert reserved==0 and mapbytes==len(mr) and 0<qn<=1<<20 and len(qr)==6344+qn and qr[512:6344]==native
    qmeta=decode(qr[6344:]);assert canon(qmeta)==qr[6344:] and qmeta==authority and hashlib.sha256(qr[6344:]).digest()==qr[264:296]
    assert dict(device=dev,inode=ino,whole_bytes=whole,mtime_ns=mtime,ctime_ns=ctime)==s['descriptor_identity']==qmeta['descriptor_identity']
    for i,k in enumerate(('map_sha256','complete_binding_sha256','capture_sha256','source_kernel_identity_sha256','native_binding_sha256')):assert qr[104+32*i:136+32*i]==bytes.fromhex(qmeta[k])
    assert qmeta['map_sha256']==h(mr) and qmeta['complete_binding_sha256']==h(canon(expected_binding)) and qmeta['native_binding_sha256']==h(native) and qmeta['capture_sha256']==h(capture) and qmeta['source_kernel_identity']==s['kernel'] and qmeta['source_kernel_identity_sha256']==h(canon(s['kernel']))
    for key,original in (('fullscan_plan_sha256','plan'),('fullscan_execution_sha256','execution'),('fullscan_release_sha256','release'),('fullscan_independent_audit_sha256','audit'),('raw_signature_sha256','signature'),('native_build_proof_sha256','build')):assert qmeta[key]==publisher.ACCEPTED[original]
    for key,filename in [('publisher_source_sha256','resident_functional_map_publish.py'),('qualification_source_sha256','resident_functional_qualification.py'),('reference_source_sha256','resident_functional_reference.py')]:assert qmeta[key]==sha(B/filename)
    for p,r in sourcepins.items():assert pin(p)==r
    put(OUT/'authority.json',authority)
    passed=True;report=dict(status='ACTUAL_FUNCTIONAL_MAP_AND_COMPANION_PUBLICATION_INDEPENDENT_BYTE_AUDIT_PASS',map=pin(OUT/'functional.ds4fmap1'),qualification=pin(OUT/'qualification.ds4fqal1'),authority=pin(OUT/'authority.json'),all640_capture_records_and_120_tensor_roles_exact=True,all3276800_FP32_probes_and3932160_FP16_signatures_finite=True,full_payloads_equal_accepted_capture_and_signature=True,independent_5832_native_binding_reconstructed=True,complete_source_kernel_identity_sha256=qmeta['source_kernel_identity_sha256'],all_accepted_ROOT_proof_identities_exact=True,no_current_model_FD_metadata_helper_guard_GPU_tokenizer=True,no_native_runtime_or_coefficient_bits_or_quality_proof=True,no_retry=True)
except Exception as e:
    report=dict(status='FIRST_ACTUAL_MAP_PUBLICATION_FAILURE_RETAINED',error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),no_retry=True)
report.update(wall_seconds_diagnostic=time.monotonic()-start,ROOT_grant_sha256=sha(OUT/'root-grant.json'));put(OUT/'root-proof.json',report);print(json.dumps({k:report[k] for k in ('status','wall_seconds_diagnostic')}));print('PROOF',sha(OUT/'root-proof.json'));raise SystemExit(0 if passed else 3)
