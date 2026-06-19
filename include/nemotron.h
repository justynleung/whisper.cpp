#ifndef NEMOTRON_H
#define NEMOTRON_H

#include "ggml.h"
#include "ggml-cpu.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __GNUC__
#    define NEMOTRON_DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define NEMOTRON_DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define NEMOTRON_DEPRECATED(func, hint) func
#endif

#ifdef NEMOTRON_SHARED
#    ifdef _WIN32
#        ifdef NEMOTRON_BUILD
#            define NEMOTRON_API __declspec(dllexport)
#        else
#            define NEMOTRON_API __declspec(dllimport)
#        endif
#    else
#        define NEMOTRON_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define NEMOTRON_API
#endif

#define NEMOTRON_SAMPLE_RATE 16000
#define NEMOTRON_HOP_LENGTH  160

#define NEMOTRON_DEFAULT_LANGUAGE_ID -1
#define NEMOTRON_PROMPT_DIM          128

#define NEMOTRON_STREAM_LEFT_CONTEXT_FRAMES 56
#define NEMOTRON_STREAM_RIGHT_CONTEXT_AUTO  -1

#ifdef __cplusplus
extern "C" {
#endif

    struct nemotron_context;
    struct nemotron_state;
    struct nemotron_stream;
    struct nemotron_full_params;

    typedef int32_t nemotron_pos;
    typedef int32_t nemotron_token;
    typedef int32_t nemotron_seq_id;

    struct nemotron_context_params {
        bool  use_gpu;
        int   gpu_device;  // CUDA device
    };

    typedef struct nemotron_token_data {
        nemotron_token id;  // the BPE subword ID

        float p;
        float plog;

        int frame_index;

        int64_t t0;
        int64_t t1;

        bool is_word_start;
    } nemotron_token_data;

    typedef struct nemotron_model_loader {
        void * context;

        size_t (*read)(void * ctx, void * output, size_t read_size);
        bool    (*eof)(void * ctx);
        void  (*close)(void * ctx);
    } nemotron_model_loader;

    NEMOTRON_API const char * nemotron_version(void);

    // Various functions for loading a ggml nemotron model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    NEMOTRON_API struct nemotron_context * nemotron_init_from_file_with_params  (const char * path_model,              struct nemotron_context_params params);
    NEMOTRON_API struct nemotron_context * nemotron_init_from_buffer_with_params(void * buffer, size_t buffer_size,    struct nemotron_context_params params);
    NEMOTRON_API struct nemotron_context * nemotron_init_with_params            (struct nemotron_model_loader * loader, struct nemotron_context_params params);

    // These are the same as the above, but the internal state of the context is not allocated automatically.
    // It is the responsibility of the caller to allocate the state using nemotron_init_state().
    NEMOTRON_API struct nemotron_context * nemotron_init_from_file_with_params_no_state  (const char * path_model,              struct nemotron_context_params params);
    NEMOTRON_API struct nemotron_context * nemotron_init_from_buffer_with_params_no_state(void * buffer, size_t buffer_size,    struct nemotron_context_params params);
    NEMOTRON_API struct nemotron_context * nemotron_init_with_params_no_state            (struct nemotron_model_loader * loader, struct nemotron_context_params params);

    NEMOTRON_API struct nemotron_state * nemotron_init_state(struct nemotron_context * ctx);

    // Frees all allocated memory
    NEMOTRON_API void nemotron_free      (struct nemotron_context * ctx);
    NEMOTRON_API void nemotron_free_state(struct nemotron_state * state);
    NEMOTRON_API void nemotron_free_params(struct nemotron_full_params * params);
    NEMOTRON_API void nemotron_free_context_params(struct nemotron_context_params * params);

    // Convert RAW PCM audio to log mel spectrogram.
    // The resulting spectrogram is stored inside the default state of the provided nemotron context.
    // Returns 0 on success
    NEMOTRON_API int nemotron_pcm_to_mel(
            struct nemotron_context * ctx,
                        const float * samples,
                                int   n_samples,
                                int   n_threads);

    NEMOTRON_API int nemotron_pcm_to_mel_with_state(
            struct nemotron_context * ctx,
              struct nemotron_state * state,
                        const float * samples,
                                int   n_samples,
                                int   n_threads);

    // This can be used to set a custom log mel spectrogram inside the default state of the provided nemotron context.
    // Use this instead of nemotron_pcm_to_mel() if you want to provide your own log mel spectrogram.
    // n_mel must be 128
    // Returns 0 on success
    NEMOTRON_API int nemotron_set_mel(
            struct nemotron_context * ctx,
                        const float * data,
                                int   n_len,
                                int   n_mel);

    NEMOTRON_API int nemotron_set_mel_with_state(
            struct nemotron_context * ctx,
              struct nemotron_state * state,
                        const float * data,
                                int   n_len,
                                int   n_mel);

    // Run the Nemotron encoder on the log mel spectrogram stored inside the default state in the provided nemotron context.
    // Make sure to call nemotron_pcm_to_mel() or nemotron_set_mel() first.
    // offset can be used to specify the offset of the first frame in the spectrogram.
    // Returns 0 on success
    NEMOTRON_API int nemotron_encode(
            struct nemotron_context * ctx,
                                int   offset,
                                int   n_threads);

    NEMOTRON_API int nemotron_encode_with_state(
            struct nemotron_context * ctx,
              struct nemotron_state * state,
                                int   offset,
                                int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens.
    // Returns a negative number on failure - the number of tokens that would have been returned.
    NEMOTRON_API int nemotron_tokenize(
            struct nemotron_context * ctx,
                        const char * text,
                    nemotron_token * tokens,
                               int   n_max_tokens);

    // Return the number of tokens in the provided text.
    // Equivalent to: -nemotron_tokenize(ctx, text, NULL, 0)
    NEMOTRON_API int nemotron_token_count(struct nemotron_context * ctx, const char * text);

    NEMOTRON_API int nemotron_n_len           (struct nemotron_context * ctx); // mel length
    NEMOTRON_API int nemotron_n_len_from_state(struct nemotron_state * state); // mel length
    NEMOTRON_API int nemotron_n_vocab         (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_n_audio_ctx     (struct nemotron_context * ctx);

    NEMOTRON_API int nemotron_model_n_vocab             (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_model_n_audio_ctx         (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_model_n_audio_state       (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_model_n_audio_head        (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_model_n_audio_layer       (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_model_n_mels              (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_model_n_subsampling_factor(struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_model_ftype               (struct nemotron_context * ctx);

    // Token logits obtained from the last call to nemotron_full/nemotron_chunk.
    // The logits for the last token are stored in the last row.
    // Rows: n_tokens
    // Cols: n_vocab
    NEMOTRON_API float * nemotron_get_logits           (struct nemotron_context * ctx);
    NEMOTRON_API float * nemotron_get_logits_from_state(struct nemotron_state * state);

    // Token Id -> String. Uses the vocabulary in the provided context.
    NEMOTRON_API const char * nemotron_token_to_str(struct nemotron_context * ctx, nemotron_token token);

    NEMOTRON_API int nemotron_token_to_text(const char * token_str, bool is_first, char * output, int max_len);

    // Special tokens
    NEMOTRON_API nemotron_token nemotron_token_blank(struct nemotron_context * ctx);
    NEMOTRON_API nemotron_token nemotron_token_unk  (struct nemotron_context * ctx);
    NEMOTRON_API nemotron_token nemotron_token_bos  (struct nemotron_context * ctx);

    // Performance information from the default state.
    struct nemotron_timings {
        float sample_ms;
        float encode_ms;
        float decode_ms;
    };
    NEMOTRON_API struct nemotron_timings * nemotron_get_timings(struct nemotron_context * ctx);
    NEMOTRON_API void nemotron_print_timings(struct nemotron_context * ctx);
    NEMOTRON_API void nemotron_reset_timings(struct nemotron_context * ctx);

    // Print system information
    NEMOTRON_API const char * nemotron_print_system_info(void);

    // Available sampling strategies
    enum nemotron_sampling_strategy {
        NEMOTRON_SAMPLING_GREEDY,
    };

    // Token callback.
    // Called for each new predicted token.
    // Use the nemotron_full_...() functions to obtain the text segments.
    typedef void (*nemotron_new_token_callback)(
            struct nemotron_context * ctx,
              struct nemotron_state * state,
          const nemotron_token_data * token_data,
                               void * user_data);

    // Text segment callback
    // Called on every newly generated text segment.
    // Use the nemotron_full_...() functions to obtain the text segments.
    typedef void (*nemotron_new_segment_callback)(struct nemotron_context * ctx, struct nemotron_state * state, int n_new, void * user_data);

    // Progress callback
    typedef void (*nemotron_progress_callback)(struct nemotron_context * ctx, struct nemotron_state * state, int progress, void * user_data);

    // Encoder begin callback
    // If not NULL, called before the encoder starts.
    // If it returns false, the computation is aborted.
    typedef bool (*nemotron_encoder_begin_callback)(struct nemotron_context * ctx, struct nemotron_state * state, void * user_data);

    // Parameters for the nemotron_full() function.
    // If you change the order or add new parameters, make sure to update the default values in nemotron.cpp:
    // nemotron_full_default_params()
    struct nemotron_full_params {
        enum nemotron_sampling_strategy strategy;

        int n_threads;
        int offset_ms;          // start offset in ms
        int duration_ms;        // audio duration to process in ms

        bool no_context;        // do not use past transcription (if any) as context

        int  audio_ctx;         // overwrite the audio context size (0 = use default)

        // Language prompt selection. NEMOTRON_DEFAULT_LANGUAGE_ID means implementation default.
        // If language is not NULL, implementation should validate it and prefer it over language_id.
        int          language_id;
        const char * language;

        // Cache-aware streaming settings in encoder frames. Supported right context values are 0, 1, 3, 6, and 13.
        // NEMOTRON_STREAM_RIGHT_CONTEXT_AUTO means implementation default.
        int stream_left_context;
        int stream_right_context;

        // called for every newly generated text segment
        nemotron_new_segment_callback new_segment_callback;
        void * new_segment_callback_user_data;

        // called for every newly generated token
        nemotron_new_token_callback new_token_callback;
        void * new_token_callback_user_data;

        // called on each progress update
        nemotron_progress_callback progress_callback;
        void * progress_callback_user_data;

        // called each time before the encoder starts
        nemotron_encoder_begin_callback encoder_begin_callback;
        void * encoder_begin_callback_user_data;

        // called each time before ggml computation starts
        ggml_abort_callback abort_callback;
        void * abort_callback_user_data;
    };

    // NOTE: this function allocates memory, and it is the responsibility of the caller to free the pointer - see nemotron_free_context_params() & nemotron_free_params()
    NEMOTRON_API struct nemotron_context_params * nemotron_context_default_params_by_ref(void);
    NEMOTRON_API struct nemotron_context_params   nemotron_context_default_params       (void);

    NEMOTRON_API struct nemotron_full_params * nemotron_full_default_params_by_ref(enum nemotron_sampling_strategy strategy);
    NEMOTRON_API struct nemotron_full_params   nemotron_full_default_params       (enum nemotron_sampling_strategy strategy);

    // Run the entire model: PCM -> log mel spectrogram -> encoder -> decoder -> text
    // Not thread safe for same context
    NEMOTRON_API int nemotron_full(
                struct nemotron_context * ctx,
            struct nemotron_full_params   params,
                            const float * samples,
                                    int   n_samples);

    NEMOTRON_API int nemotron_full_with_state(
                struct nemotron_context * ctx,
                  struct nemotron_state * state,
            struct nemotron_full_params   params,
                            const float * samples,
                                    int   n_samples);

    // Process a single chunk of audio data that fits within the model's cache-aware window.
    NEMOTRON_API int nemotron_chunk(
                struct nemotron_context * ctx,
                  struct nemotron_state * state,
            struct nemotron_full_params   params,
                            const float * samples,
                                   int    n_samples);

    // Streaming API. Push accepts raw PCM samples and may emit callbacks according to params.
    NEMOTRON_API struct nemotron_stream * nemotron_stream_init(
                struct nemotron_context * ctx,
                  struct nemotron_state * state,
            struct nemotron_full_params   params);

    NEMOTRON_API int nemotron_stream_push(
                 struct nemotron_stream * stream,
                            const float * samples,
                                    int   n_samples);

    NEMOTRON_API int nemotron_stream_flush(struct nemotron_stream * stream);
    NEMOTRON_API void nemotron_stream_reset(struct nemotron_stream * stream);
    NEMOTRON_API void nemotron_stream_free (struct nemotron_stream * stream);

    // Number of generated text segments
    NEMOTRON_API int nemotron_full_n_segments           (struct nemotron_context * ctx);
    NEMOTRON_API int nemotron_full_n_segments_from_state(struct nemotron_state * state);

    // Get the start and end time of the specified segment
    NEMOTRON_API int64_t nemotron_full_get_segment_t0           (struct nemotron_context * ctx, int i_segment);
    NEMOTRON_API int64_t nemotron_full_get_segment_t0_from_state(struct nemotron_state * state, int i_segment);

    NEMOTRON_API int64_t nemotron_full_get_segment_t1           (struct nemotron_context * ctx, int i_segment);
    NEMOTRON_API int64_t nemotron_full_get_segment_t1_from_state(struct nemotron_state * state, int i_segment);

    // Get the text of the specified segment
    NEMOTRON_API const char * nemotron_full_get_segment_text           (struct nemotron_context * ctx, int i_segment);
    NEMOTRON_API const char * nemotron_full_get_segment_text_from_state(struct nemotron_state * state, int i_segment);

    // Get number of tokens in the specified segment
    NEMOTRON_API int nemotron_full_n_tokens           (struct nemotron_context * ctx, int i_segment);
    NEMOTRON_API int nemotron_full_n_tokens_from_state(struct nemotron_state * state, int i_segment);

    // Get the token text of the specified token in the specified segment
    NEMOTRON_API const char * nemotron_full_get_token_text           (struct nemotron_context * ctx, int i_segment, int i_token);
    NEMOTRON_API const char * nemotron_full_get_token_text_from_state(struct nemotron_context * ctx, struct nemotron_state * state, int i_segment, int i_token);

    // Get the token id of the specified token in the specified segment
    NEMOTRON_API nemotron_token nemotron_full_get_token_id           (struct nemotron_context * ctx, int i_segment, int i_token);
    NEMOTRON_API nemotron_token nemotron_full_get_token_id_from_state(struct nemotron_state * state, int i_segment, int i_token);

    // Get token data for the specified token in the specified segment
    NEMOTRON_API nemotron_token_data nemotron_full_get_token_data           (struct nemotron_context * ctx, int i_segment, int i_token);
    NEMOTRON_API nemotron_token_data nemotron_full_get_token_data_from_state(struct nemotron_state * state, int i_segment, int i_token);

    // Get the probability of the specified token in the specified segment
    NEMOTRON_API float nemotron_full_get_token_p           (struct nemotron_context * ctx, int i_segment, int i_token);
    NEMOTRON_API float nemotron_full_get_token_p_from_state(struct nemotron_state * state, int i_segment, int i_token);

    // Control logging output; default behavior is to print to stderr
    NEMOTRON_API void nemotron_log_set(ggml_log_callback log_callback, void * user_data);

#ifdef __cplusplus
}
#endif

#endif
