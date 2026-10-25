/* Narrow FFmpeg memory/codec boundary. Model preprocessing remains native Zig. */
#include "media_codec.h"
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/imgutils.h>
#include <libavutil/mathematics.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const uint8_t *bytes; size_t length, position; } memory_input;
static int read_memory(void *opaque, uint8_t *output, int capacity) {
    memory_input *input=opaque;
    size_t left=input->length-input->position;
    if (!left) return AVERROR_EOF;
    size_t count=left<(size_t)capacity?left:(size_t)capacity;
    memcpy(output,input->bytes+input->position,count);
    input->position+=count;
    return (int)count;
}
static int64_t seek_memory(void *opaque, int64_t offset, int whence) {
    memory_input *input=opaque;
    if (whence==AVSEEK_SIZE) return (int64_t)input->length;
    whence&=~AVSEEK_FORCE;
    int64_t base=whence==SEEK_SET?0:whence==SEEK_CUR?(int64_t)input->position:whence==SEEK_END?(int64_t)input->length:-1;
    if (base<0 || offset < -base || offset > (int64_t)input->length-base) return AVERROR(EINVAL);
    input->position=(size_t)(base+offset);
    return (int64_t)input->position;
}
void ds4_clef_media_free(ds4_clef_media *media) {
    if (!media) return;
    for (size_t i=0;i<media->frame_count;i++) free(media->frames[i].rgb);
    free(media->frames);free(media->samples);
    memset(media,0,sizeof(*media));
}
static int fail(char *text,size_t capacity,const char *message,int code) {
    char detail[AV_ERROR_MAX_STRING_SIZE];
    if (code<0) { av_strerror(code,detail,sizeof(detail));snprintf(text,capacity,"%s: %s",message,detail); }
    else snprintf(text,capacity,"%s",message);
    return 0;
}
static AVCodecContext *decoder(AVFormatContext *format,int stream,const AVCodec *codec) {
    AVCodecContext *context=avcodec_alloc_context3(codec);
    if (!context) return NULL;
    context->thread_count=2;
    context->max_pixels=64LL*1024*1024;
    if (avcodec_parameters_to_context(context,format->streams[stream]->codecpar)<0 || avcodec_open2(context,codec,NULL)<0) avcodec_free_context(&context);
    return context;
}
static int grow_audio(ds4_clef_media *output,size_t *capacity,size_t count,size_t maximum) {
    if (count>maximum-output->sample_count) return 0;
    const size_t wanted=output->sample_count+count;
    if (wanted<=*capacity) return 1;
    size_t next=*capacity?*capacity:(maximum<16384?maximum:16384);
    while (next<wanted) next=next>maximum/2?maximum:next*2;
    float *samples=realloc(output->samples,next*sizeof(*samples));
    if (!samples) return 0;
    output->samples=samples;*capacity=next;
    return 1;
}
static int audio_frame(ds4_clef_media *output,AVFrame *frame,SwrContext **resampler,
                       size_t *capacity,size_t maximum,AVRational timebase) {
    if (!*resampler) {
        AVChannelLayout mono=AV_CHANNEL_LAYOUT_MONO;
        if (frame->sample_rate<=0 || frame->ch_layout.nb_channels<=0 ||
            swr_alloc_set_opts2(resampler,&mono,AV_SAMPLE_FMT_FLT,16000,&frame->ch_layout,
                               (enum AVSampleFormat)frame->format,frame->sample_rate,0,NULL)<0 || swr_init(*resampler)<0) return 0;
        if (frame->best_effort_timestamp!=AV_NOPTS_VALUE) output->audio_start_seconds=frame->best_effort_timestamp*av_q2d(timebase);
    }
    int needed=(int)av_rescale_rnd(swr_get_delay(*resampler,frame->sample_rate)+frame->nb_samples,
                                  16000,frame->sample_rate,AV_ROUND_UP);
    if (needed<0 || !grow_audio(output,capacity,(size_t)needed,maximum)) return 0;
    uint8_t *target=(uint8_t*)(output->samples+output->sample_count);
    int count=swr_convert(*resampler,&target,needed,(const uint8_t**)frame->extended_data,frame->nb_samples);
    if (count<0) return 0;
    output->sample_count+=(size_t)count;
    return 1;
}
static int video_frame(ds4_clef_media *output,AVFrame *frame,struct SwsContext **scaler,
                       size_t maximum,size_t pixels,double target_fps,AVRational timebase) {
    double seconds=frame->best_effort_timestamp==AV_NOPTS_VALUE?
        output->frame_count/target_fps:frame->best_effort_timestamp*av_q2d(timebase);
    // Container timestamps may start above or below zero. Sample relative to
    // the first decoded frame, retaining the original times for audio alignment.
    const double first=output->frame_count?output->frames[0].seconds:seconds;
    if (seconds+1e-6 < first+output->frame_count/target_fps) return 1;
    if (output->frame_count>=maximum || frame->width<=0 || frame->height<=0) return 0;
    double scale=fmin(1.0,sqrt((double)pixels/((double)frame->width*frame->height)));
    int width=(int)fmax(2,rint(frame->width*scale/2)*2);
    int height=(int)fmax(2,rint(frame->height*scale/2)*2);
    ds4_clef_media_frame *frames=realloc(output->frames,(output->frame_count+1)*sizeof(*frames));
    if (!frames) return 0;
    output->frames=frames;
    uint8_t *rgb=malloc((size_t)width*height*3);
    if (!rgb) return 0;
    *scaler=sws_getCachedContext(*scaler,frame->width,frame->height,(enum AVPixelFormat)frame->format,
                               width,height,AV_PIX_FMT_RGB24,SWS_BILINEAR,NULL,NULL,NULL);
    if (!*scaler) {free(rgb);return 0;}
    uint8_t *dest[4]={rgb,NULL,NULL,NULL};int lines[4]={width*3,0,0,0};
    if (sws_scale(*scaler,(const uint8_t*const*)frame->data,frame->linesize,0,frame->height,dest,lines)!=height) {free(rgb);return 0;}
    output->frames[output->frame_count++]=(ds4_clef_media_frame){rgb,width,height,seconds};
    return 1;
}
int ds4_clef_media_decode(const uint8_t *bytes,size_t length,int video,double target_fps,
                         size_t max_frames,size_t max_samples,size_t max_frame_pixels,
                         ds4_clef_media *out,char *error,size_t error_capacity) {
    memset(out,0,sizeof(*out));
    if (error_capacity) error[0]=0;
    if (!bytes || !length || length>INT64_MAX || !max_samples || (video && (!max_frames || !max_frame_pixels || target_fps<=0))) return fail(error,error_capacity,"invalid media limits",0);
    memory_input input={bytes,length,0};
    AVFormatContext *format=avformat_alloc_context();
    AVIOContext *io=NULL;AVCodecContext *audio=NULL,*vision=NULL;
    AVFrame *frame=NULL;AVPacket *packet=NULL;SwrContext *resampler=NULL;struct SwsContext *scaler=NULL;
    size_t audio_capacity=0;int success=0;
    uint8_t *buffer=av_malloc(32768);
    if (!format || !buffer) {av_free(buffer);goto done;}
    io=avio_alloc_context(buffer,32768,0,&input,read_memory,NULL,seek_memory);
    if (!io) {av_free(buffer);goto done;}
    format->pb=io;format->flags|=AVFMT_FLAG_CUSTOM_IO;
    /* A nested playlist must not turn already-fetched bytes into file/network I/O. */
    format->protocol_whitelist=av_strdup("");
    int code=avformat_open_input(&format,NULL,NULL,NULL);
    if (code<0) {fail(error,error_capacity,"cannot open encoded media",code);goto done;}
    code=avformat_find_stream_info(format,NULL);
    if (code<0) {fail(error,error_capacity,"cannot read media streams",code);goto done;}
    const AVCodec *audio_codec=NULL,*video_codec=NULL;
    int ai=av_find_best_stream(format,AVMEDIA_TYPE_AUDIO,-1,-1,&audio_codec,0);
    int vi=video?av_find_best_stream(format,AVMEDIA_TYPE_VIDEO,-1,-1,&video_codec,0):-1;
    if ((video && vi<0) || (!video && ai<0)) {fail(error,error_capacity,"requested media stream is absent",0);goto done;}
    if (ai>=0) audio=decoder(format,ai,audio_codec);
    if (vi>=0) vision=decoder(format,vi,video_codec);
    if ((ai>=0 && !audio) || (vi>=0 && !vision)) {fail(error,error_capacity,"cannot initialize media decoder",0);goto done;}
    if (vi>=0) out->source_fps=av_q2d(av_guess_frame_rate(format,format->streams[vi],NULL));
    frame=av_frame_alloc();packet=av_packet_alloc();
    if (!frame || !packet) goto done;
    int flushing=0;
    while (!flushing) {
        code=av_read_frame(format,packet);
        if (code<0 && code!=AVERROR_EOF) {fail(error,error_capacity,"cannot read media packet",code);goto done;}
        flushing=code==AVERROR_EOF;
        for (int pass=0;pass<2;pass++) {
            int stream=pass?vi:ai;
            AVCodecContext *context=pass?vision:audio;
            if (!context || (!flushing && packet->stream_index!=stream)) continue;
            code=avcodec_send_packet(context,flushing?NULL:packet);
            if (code<0 && code!=AVERROR_EOF) {fail(error,error_capacity,"cannot send media packet",code);goto done;}
            while ((code=avcodec_receive_frame(context,frame))>=0) {
                int ok=pass?video_frame(out,frame,&scaler,max_frames,max_frame_pixels,target_fps,format->streams[stream]->time_base):
                    audio_frame(out,frame,&resampler,&audio_capacity,max_samples,format->streams[stream]->time_base);
                av_frame_unref(frame);
                if (!ok) {fail(error,error_capacity,"media conversion failed or request capacity exceeded",0);goto done;}
            }
            if (code!=AVERROR(EAGAIN) && code!=AVERROR_EOF) {fail(error,error_capacity,"cannot decode media frame",code);goto done;}
        }
        av_packet_unref(packet);
    }
    if (resampler) {
        const int pending=(int)swr_get_delay(resampler,16000);
        if (pending>0) {
            if (!grow_audio(out,&audio_capacity,(size_t)pending,max_samples)) {fail(error,error_capacity,"audio sample limit exceeded",0);goto done;}
            uint8_t *target=(uint8_t*)(out->samples+out->sample_count);
            int count=swr_convert(resampler,&target,pending,NULL,0);
            if (count<0) goto done;
            out->sample_count+=(size_t)count;
        }
    }
    if ((video && !out->frame_count) || (!video && !out->sample_count)) {fail(error,error_capacity,"no decoded media",0);goto done;}
    success=1;
 done:
    sws_freeContext(scaler);swr_free(&resampler);
    av_frame_free(&frame);av_packet_free(&packet);
    avcodec_free_context(&audio);avcodec_free_context(&vision);
    avformat_close_input(&format);
    if (io) {av_freep(&io->buffer);avio_context_free(&io);}
    if (!success) {ds4_clef_media_free(out);if (error_capacity && !error[0]) snprintf(error,error_capacity,"media decode allocation/conversion failed");}
    return success;
}
