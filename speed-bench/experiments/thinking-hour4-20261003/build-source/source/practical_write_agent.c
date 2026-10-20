/* Private literal legacy frontend/public ABI; proposed, uncompiled caller. */
#define main pt_original_main
#include "source/instrumented_agent.c"
#undef main
_Static_assert(sizeof(int)==4 && sizeof(float)==4,"fixed arm64 i32/FP32 artifacts");
int main(int argc,char **argv) {
    pt_init(); pt_write_init();
    return pt_finish(pt_write_finish(pt_original_main(argc,argv)));
}
