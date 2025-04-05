using Microsoft.AspNetCore.Components.Forms;
using MudBlazor;
using System.Runtime.Serialization;
using System.Text.Json;

namespace DiffusionPipeInterface.ViewModels
{
    public class DatasetDirectoryConfigurationViewModel
    {
        [IgnoreDataMember]
        public string Name { get; set; } = null!;

        [DataMember(Name = "path")]
        public string Path { get; set; } = null!;

        [DataMember(Name = "mask_path")]
        public string? MaskPath { get; set; }

        [DataMember(Name = "num_repeats")]
        public int NumRepeats { get; set; } = 1;

        [IgnoreDataMember]
        [DataMember(Name = "enable_ar_bucket")]
        public bool EnableARBucket { get; set; } = true;

        [IgnoreDataMember]
        private string? _arBucketsJson;

        [DataMember(Name = "ar_buckets")]
        public float[]? ARBuckets
        {
            get => !string.IsNullOrEmpty(_arBucketsJson) ? JsonSerializer.Deserialize<float[]>(_arBucketsJson) : null;
            set => _arBucketsJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [IgnoreDataMember]
        public string? ARBucketsJson
        {
            get => _arBucketsJson;
            set => _arBucketsJson = value;
        }

        [IgnoreDataMember]
        private string? _resolutionsJson;

        [DataMember(Name = "resolutions")]
        public int[]? Resolutions
        {
            get => !string.IsNullOrEmpty(_resolutionsJson) ? JsonSerializer.Deserialize<int[]>(_resolutionsJson) : null;
            set => _resolutionsJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [IgnoreDataMember]
        public string? ResolutionsJson
        {
            get => _resolutionsJson;
            set => _resolutionsJson = value;
        }

        [IgnoreDataMember]
        private string? _frameBucketsJson;

        [DataMember(Name = "frame_buckets")]
        public int[]? FrameBuckets
        {
            get => !string.IsNullOrEmpty(_frameBucketsJson) ? JsonSerializer.Deserialize<int[]>(_frameBucketsJson) : null;
            set => _frameBucketsJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [IgnoreDataMember]
        public string? FrameBucketsJson
        {
            get => _frameBucketsJson;
            set => _frameBucketsJson = value;
        }

        [IgnoreDataMember]
        [DataMember(Name = "min_ar")]
        public float MinAR { get; set; } = 0.5f;

        [IgnoreDataMember]
        [DataMember(Name = "max_ar")]
        public float MaxAR { get; set; } = 2.0f;

        [IgnoreDataMember]
        [DataMember(Name = "num_ar_buckets")]
        public int NumARBuckets { get; set; } = 7;

        [IgnoreDataMember]
        public bool DisableRemove { get; set; } = false;


        [IgnoreDataMember]
        public int CurrentUploadTotalFiles { get; set; } = 0;

        [IgnoreDataMember]
        public List<string>? TotalFilesUploaded { get; set; } = null;



        //[IgnoreDataMember]
        //public IReadOnlyList<IBrowserFile>? Files { get; set; }

        [IgnoreDataMember]
        public MudFileUpload<IReadOnlyList<IBrowserFile>?> MudFileUploadRef { get; set; } = null!;
    }
}
