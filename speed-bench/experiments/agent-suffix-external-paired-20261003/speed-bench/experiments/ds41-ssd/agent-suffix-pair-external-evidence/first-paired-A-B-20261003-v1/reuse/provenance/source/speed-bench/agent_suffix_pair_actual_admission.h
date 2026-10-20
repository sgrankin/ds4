/* Observation only: selector reservations are not allocated native undo or
 * a measured global application peak. Existing admission math is unchanged. */
#ifndef AGENT_SUFFIX_PAIR_ACTUAL_ADMISSION_H
#define AGENT_SUFFIX_PAIR_ACTUAL_ADMISSION_H
static bool pair_actual_admission_record(const char *stage,bool first,uint64_t owned,
    uint64_t text,uint64_t splice,uint64_t undo,uint64_t checkpoint,uint64_t reserve){
    uint64_t terminator=undo?1:0,temporary=text;
    if(splice>UINT64_MAX-temporary)return false;temporary+=splice;
    if(undo>UINT64_MAX-temporary)return false;temporary+=undo;
    if(terminator>UINT64_MAX-temporary)return false;temporary+=terminator;
    if(checkpoint>UINT64_MAX-temporary)return false;temporary+=checkpoint;
    char path[2048];if(!pair_actual_path(path,sizeof(path),"admission-ledger.jsonl"))return false;
    FILE *f=fopen(path,first?"wx":"a");if(!f)return false;
    bool ok=fprintf(f,"{\"status\":\"SELECTOR_FORECAST_RESERVATION_CHECK\",\"stage\":\"%s\",\"worker_owned_bytes\":%llu,\"text_requested_bytes\":%llu,\"text_allocated\":%s,\"splice_forecast_bytes\":%llu,\"canonical_undo_forecast_bytes\":%llu,\"undo_terminator_forecast_bytes\":%llu,\"checkpoint_token_forecast_bytes\":%llu,\"temporary_forecast_bytes\":%llu,\"scratch_reserved_bytes\":%llu,\"budget_bytes\":%llu,\"admitted\":%s,\"canonical_undo_allocated_by_selector\":false,\"observed_global_peak\":false}\n",
        stage,(unsigned long long)owned,(unsigned long long)text,first?"false":"true",
        (unsigned long long)splice,(unsigned long long)undo,(unsigned long long)terminator,
        (unsigned long long)checkpoint,(unsigned long long)temporary,(unsigned long long)reserve,
        (unsigned long long)PAIR_ACTUAL_BUDGET,pair_actual_admit(owned,temporary,reserve)?"true":"false")>=0;
    if(fclose(f))ok=false;return ok;
}
#endif
