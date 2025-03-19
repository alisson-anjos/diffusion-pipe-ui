using DiffusionPipeInterface.Components;
using MudBlazor;

namespace DiffusionPipeInterface.Utils
{
    public static class Tooltips
    {
        public const string BlocksToSwap = @"This value controls the number of blocks kept offloaded to RAM. 
Increasing it lowers VRAM use, but has a performance penalty. 
The exactly performance penalty depends on the model and the type of training you are doing (e.g. images vs video).
Recommended Value: 20";

        public const string MicroBatchSizePerGpu = @"Batch size of a single forward/backward pass for one GPU.";

        public const string GradientAccumulationSteps = @"If pipeline_stages > 1, a higher GAS means better GPU utilization due to smaller pipeline bubbles (where GPUs aren't overlapping computation).";

        public const string SaveEveryNEpochs = @"Probably want to set this a bit higher if you have a smaller dataset so you don't end up with a million saved models.";

        public const string ActivationCheckpointing = @"Always set to true unless you have a huge amount of VRAM.";

        public const string PartitionMethod = @"Controls how Deepspeed decides how to divide layers across GPUs. Probably don't change this.";

        public const string SaveDtype = @"dtype for saving the LoRA or model, if different from training dtype";

        public const string CachingBatchSize = @"Batch size for caching latents and text embeddings. 
Increasing can lead to higher GPU utilization during caching phase but uses more memory.";

        public const string StepsPerPrint = @"How often deepspeed logs to console.";

        public const string VideoClipMode = @"How to extract video clips for training from a single input video file.
The video file is first assigned to one of the configured frame buckets, but then we must extract one or more clips of exactly the right
number of frames for that bucket.
single_beginning: one clip starting at the beginning of the video
single_middle: one clip from the middle of the video (cutting off the start and end equally)
multiple_overlapping: extract the minimum number of clips to cover the full range of the video. They might overlap some.
default is single_middle";

        public const string ModelDtype = @"Base dtype used for all models.";

        public const string TransformerDtype = @"Supports fp8 for the transformer when training LoRA.";

        public const string TimestepSampleMethod = @"How to sample timesteps to train on. Can be logit_normal or uniform.";

        public const string AdapterDtype = @"Dtype for the LoRA weights you are training.";

        public const string InitFromExisting = @"You can initialize the lora weights from a previously trained lora.";

        public const string OptimizerType = @"AdamW from the optimi library is a good default since it automatically uses Kahan summation when training bfloat16 weights.";

        public const string ChromaTransformerDtype = @"You can optionally load the transformer in fp8 when training LoRAs.";

        public const string FluxShift = @"Resolution-dependent timestep shift towards more noise. Same meaning as sd-scripts.";

        public const string BypassGuidanceEmbedding = @"For FLEX.1-alpha, you can bypass the guidance embedding which is the recommended way to train that model.";

        public const string VPred = @"You can train v-prediction models (e.g. NoobAI vpred) by setting this option.";

        public const string MinSnrGamma = @"Min SNR is supported. Same meaning as sd-scripts";

        public const string DebiasedEstimationLoss = @"Debiased estimation loss is supported. Same meaning as sd-scripts.";

        public const string UnetLr = @"You can set separate learning rates for unet and text encoders. If one of these isn't set, the optimizer learning rate will apply.";

        public const string SingleFilePath = @"Point this to one of the single checkpoint files to load the transformer and VAE from it.";

        public const string LuminaShift = @"";

        public const string Epochs = @"I usually set this to a really high value because I don't know how long I want to train.";

        public const string Resolutions = @"Resolutions to train on, given as the side length of a square image. You can have multiple sizes here.
!!!WARNING!!!: this might work differently to how you think it does. Images are first grouped to aspect ratio
buckets, then each image is resized to ALL of the areas specified by the resolutions list. This is a way to do
multi-resolution training, i.e. training on multiple total pixel areas at once. Your dataset is effectively duplicated
as many times as the length of this list.

You can give resolutions as (width, height) pairs also. This doesn't do anything different, 
it's just another way of specifying the area(s) (i.e. total number of pixels) you want to train on.
[[1280, 720]]
";

        public const string FrameBuckets = @"For video training, you need to configure frame buckets (similar to aspect ratio buckets). There will always
be a frame bucket of 1 for images. Videos will be assigned to the longest frame bucket possible, such that the video
is still greater than or equal to the frame bucket length.
But videos are never assigned to the image frame bucket (1); if the video is very short it would just be dropped.

If you have >24GB VRAM, or multiple GPUs and use pipeline parallelism, or lower the spatial resolution, you could maybe train with longer frame buckets
[1, 33, 65, 97]";

        public const string EnableArBuckets = @"Enable aspect ratio bucketing. For the different AR buckets, the final size will be such that
the areas match the resolutions you configured.";

        public const string MinMaxAr = @"Min and max aspect ratios, given as width/height ratio.";

        public const string ArBuckets = @"Can manually specify ar_buckets instead of using the range-style configured.
Each entry can be width/height ratio, or (width, height) pair. But you can't mix them, because of TOML.

ar_buckets = [[512, 512], [448, 576]]
ar_buckets = [1.0, 1.5]
";

        public const string NumArBuckets = @"Total number of aspect ratio buckets, evenly spaced (in log space) between min_ar and max_ar.";

        public const string NumRepeats = @"How many repeats for 1 epoch. The dataset will act like it is duplicated this many times.
The semantics of this are the same as sd-scripts: num_repeats=1 means one epoch is a single pass over all examples (no duplication).";

        public const string ResolutionsSubDataset = @"Overrides the resolution set in the general settings.";

        public const string ArBucketsSubDataset = @"Overrides the AR Buckets set in the general settings.";

        public const string FrameBucketsSubDataset = @"Overrides the Frame Buckets set in the general settings.";
    }
}
