/* No generation: exercise the actual CLI transcript updater with the V4.1 fixture. */
#define main ds4_cli_main
#include "../ds4_cli.c"
#undef main
#include <assert.h>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    ds4_engine *engine = NULL;
    ds4_engine_options opt = {.model_path = argv[1], .backend = DS4_BACKEND_METAL,
        .ssd_streaming = true, .ssd_streaming_cache_experts = 512,
        .context_size = 256, .power_percent = 100};
    assert(ds4_engine_open(&engine, &opt) == 0);
    const ds4_think_mode modes[] = {DS4_THINK_NONE, DS4_THINK_HIGH,
        (ds4_think_mode)(DS4_THINK_LEVEL_BASE + 25), DS4_THINK_MAX,
        (ds4_think_mode)DS4_THINK_LEVEL_BASE, DS4_THINK_HIGH, DS4_THINK_NONE};
    for (int system = 0; system < 2; system++) {
        repl_chat chat = {.initial_system_text = system != 0};
        ds4_chat_begin(engine, &chat.transcript);
        chat.think_prefix_pos = chat.transcript.len;
        ds4_cli_chat_apply_think_prefix(engine, &chat, modes[0]);
        if (system) ds4_chat_append_message(engine, &chat.transcript, "system", "System text.");
        ds4_chat_append_message(engine, &chat.transcript, "user", "First question");
        ds4_chat_append_message(engine, &chat.transcript, "assistant", "First answer");
        ds4_vision_span *image = ds4_cli_chat_add_image(&chat);
        assert(image && chat.image_count == 1);
        for (size_t i = 0; i < sizeof(*image); i++)
            assert(((const unsigned char *)image)[i] == 0);
        image->token_start = (uint32_t)chat.transcript.len - 1;
        for (size_t i = 0; i < sizeof(modes) / sizeof(*modes); i++) {
            const int old_prefix = chat.think_prefix_tokens;
            const uint32_t old_image_start = image->token_start;
            assert(ds4_cli_chat_apply_think_prefix(engine, &chat, modes[i]));
            assert(image->token_start == (uint32_t)((int64_t)old_image_start +
                   chat.think_prefix_tokens - old_prefix));
            ds4_tokens expected = {0};
            ds4_chat_begin(engine, &expected);
            ds4_chat_append_think_prefix(engine, &expected, modes[i]);
            if (system) ds4_chat_append_message(engine, &expected, "system", "System text.");
            ds4_chat_append_message(engine, &expected, "user", "First question");
            ds4_chat_append_message(engine, &expected, "assistant", "First answer");
            assert(expected.len == chat.transcript.len &&
                   ds4_tokens_starts_with(&chat.transcript, &expected));
            ds4_tokens_free(&expected);
        }
        ds4_tokens before = {0};
        ds4_tokens_copy(&before, &chat.transcript);
        ds4_vision_span *images_before = chat.images;
        const size_t image_count_before = chat.image_count, image_cap_before = chat.image_cap;
        unsigned char image_before[sizeof(*image)];
        memcpy(image_before, image, sizeof(image_before));
        chat.ctx_size = chat.transcript.len + 1;
        assert(!ds4_cli_chat_apply_think_prefix(engine, &chat, DS4_THINK_MAX));
        assert(before.len == chat.transcript.len && ds4_tokens_starts_with(&chat.transcript, &before));
        assert(chat.images == images_before && chat.image_count == image_count_before &&
               chat.image_cap == image_cap_before && !memcmp(image, image_before, sizeof(image_before)));
        ds4_tokens_free(&before);
        ds4_cli_chat_free(&chat);
    }
    ds4_engine_close(engine);
    puts("V4.1 CLI thinking prefix transitions and image offsets: PASS");
    return 0;
}
