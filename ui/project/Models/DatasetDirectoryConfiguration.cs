using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Text.Json;

namespace DiffusionPipeInterface.Models
{
    public class DatasetDirectoryConfiguration
    {
        [Key]
        public int Id { get; set; }

        public string Path { get; set; } = null!;
        public string? MaskPath { get; set; }
        public int NumRepeats { get; set; } = 10;
        //public bool EnableARBucket { get; set; } = true;

        private string? _arBucketsJson;

        [NotMapped]
        public float[]? ARBuckets
        {
            get => _arBucketsJson != null && !string.IsNullOrEmpty(_arBucketsJson) ? JsonSerializer.Deserialize<float[]>(_arBucketsJson) : null;
            set => _arBucketsJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [Column("ARBuckets")]
        public string? ARBucketsJson
        {
            get => _arBucketsJson;
            set => _arBucketsJson = value;
        }

        private string? _resolutionsJson;

        [NotMapped]
        public int[]? Resolutions
        {
            get => _resolutionsJson != null ? JsonSerializer.Deserialize<int[]>(_resolutionsJson) : null;
            set => _resolutionsJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [Column("Resolutions")]
        public string? ResolutionsJson
        {
            get => _resolutionsJson;
            set => _resolutionsJson = value;
        }

        private string? _frameBucketsJson;

        [NotMapped]
        public int[]? FrameBuckets
        {
            get => _frameBucketsJson != null ? JsonSerializer.Deserialize<int[]>(_frameBucketsJson) : null;
            set => _frameBucketsJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [Column("FrameBuckets")]
        public string? FrameBucketsJson
        {
            get => _frameBucketsJson;
            set => _frameBucketsJson = value;
        }

        public int DatasetConfigurationId { get; set; }
        //public float MinAR { get; set; } = 0.5f;
        //public float MaxAR { get; set; } = 2.0f;
        //public int NumARBuckets { get; set; } = 7;

        public virtual DatasetConfiguration? DatasetConfiguration { get; set; }
    }
}
