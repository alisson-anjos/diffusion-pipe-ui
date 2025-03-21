using DiffusionPipeInterface.Enums;
using System.Runtime.Serialization;
using System.Text.Json;

namespace DiffusionPipeInterface.ViewModels
{
    public class DatasetConfigurationViewModel
    {
        public string Name { get; set; } = null!;
        public ModelType Model { get; set; }

        [IgnoreDataMember]
        private string _resolutionsJson = string.Empty;

        [DataMember(Name = "resolutions")]
        public int[] Resolutions
        {
            get => JsonSerializer.Deserialize<int[]>(_resolutionsJson) ?? Array.Empty<int>();
            set => _resolutionsJson = JsonSerializer.Serialize(value);
        }

        public int[][] ResolutionsMatriz
        {
            get => JsonSerializer.Deserialize<int[][]>(_resolutionsJson) ?? Array.Empty<int[]>();
            set => _resolutionsJson = JsonSerializer.Serialize(value);
        }

        [IgnoreDataMember]
        public string ResolutionsJson
        {
            get => _resolutionsJson;
            set => _resolutionsJson = value;
        }

        [DataMember(Name = "enable_ar_bucket")]
        public bool EnableARBucket { get; set; } = true;

        [DataMember(Name = "min_ar")]
        public float MinAR { get; set; } = 0.5f;

        [DataMember(Name = "max_ar")]
        public float MaxAR { get; set; } = 2.0f;

        [DataMember(Name = "num_ar_buckets")]
        public int NumARBuckets { get; set; } = 7;

        [IgnoreDataMember]
        private string? _arBucketsJson = string.Empty;

        [DataMember(Name = "ar_buckets")]
        //[IgnoreDataMember]
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
        private string _frameBucketsJson = string.Empty;

        //[IgnoreDataMember]
        [DataMember(Name = "frame_buckets")]
        public int[] FrameBuckets
        {
            get => JsonSerializer.Deserialize<int[]>(_frameBucketsJson) ?? new int[1] { 1 };
            set => _frameBucketsJson = JsonSerializer.Serialize(value);
        }

        [IgnoreDataMember]
        public string FrameBucketsJson
        {
            get => _frameBucketsJson;
            set => _frameBucketsJson = value;
        }

        //[DataMember(Name = "directory")]
        [IgnoreDataMember]
        public virtual List<DatasetDirectoryConfigurationViewModel>? Directories { get; set; }
    }
}
