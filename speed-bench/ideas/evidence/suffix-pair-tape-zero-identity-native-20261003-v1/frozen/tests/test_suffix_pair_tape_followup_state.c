/* Actual private state comparator fixtures; no engine/model/GPU calls. */
#define main ds4_pair_followup_unused_main
#include "../speed-bench/suffix_pair_tape_followup_bench.c"
#undef main
#define REQUIRE(x) do {if(!(x)){fprintf(stderr,"state-fixture:%d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(int argc,char **argv){
    if(argc!=2)return 2;
    const uint64_t size=70000;
    ds4_session_snapshot source={.ptr=malloc(size),.len=size,.cap=size};
    ds4_session_snapshot observed={.ptr=malloc(size),.len=size,.cap=size};
    REQUIRE(source.ptr&&observed.ptr);
    for(uint64_t i=0;i<size;i++)source.ptr[i]=(uint8_t)(i*17u+31u);
    memcpy(source.ptr,"DSV4",4);memcpy(observed.ptr,source.ptr,size);
    REQUIRE(save_bytes(argv[1],".canonical.bin",&source));
    REQUIRE(state_file_equal(argv[1],".canonical.bin",&observed,false));
    observed.ptr[32770]^=1;REQUIRE(!state_file_equal(argv[1],".canonical.bin",&observed,false));observed.ptr[32770]^=1;
    memcpy(observed.ptr,"DSSD",4);
    REQUIRE(!state_file_equal(argv[1],".canonical.bin",&observed,false));
    REQUIRE(state_file_equal(argv[1],".canonical.bin",&observed,true));
    observed.ptr[32770]^=1;REQUIRE(!state_file_equal(argv[1],".canonical.bin",&observed,true));observed.ptr[32770]^=1;
    observed.ptr[size-1]^=1;REQUIRE(!state_file_equal(argv[1],".canonical.bin",&observed,true));observed.ptr[size-1]^=1;
    memcpy(observed.ptr,"NOPE",4);REQUIRE(!state_file_equal(argv[1],".canonical.bin",&observed,true));
    memcpy(observed.ptr,"DSSD",4);source.ptr[0]='X';REQUIRE(save_bytes(argv[1],".bad-native-magic.bin",&source));source.ptr[0]='D';
    REQUIRE(!state_file_equal(argv[1],".bad-native-magic.bin",&observed,true));
    source.len=size-1;REQUIRE(save_bytes(argv[1],".truncated.bin",&source));source.len=size;
    REQUIRE(!state_file_equal(argv[1],".truncated.bin",&observed,true));
    observed.len=size-1;REQUIRE(!state_file_equal(argv[1],".canonical.bin",&observed,true));observed.len=size;
    observed.len=3;REQUIRE(!state_file_equal(argv[1],".canonical.bin",&observed,true));
    free(source.ptr);free(observed.ptr);
    puts("follow-up actual state comparator fixtures PASS; no engine/model/GPU calls");return 0;
}
