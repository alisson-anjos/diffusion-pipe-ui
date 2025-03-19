using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.Models;
using DiffusionPipeInterface.Models.Models;
using DiffusionPipeInterface.Models.ViewModels;
using DiffusionPipeInterface.ViewModels;
using DiffusionPipeInterface.ViewModels.Models;
using System.ComponentModel;
using System.Globalization;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text.Json;
using Tomlyn;
using Tomlyn.Model;

namespace DiffusionPipeInterface.Utils
{
    public static class Extensions
    {
        public static string GetDescription(this Enum enumValue)
        {
            var field = enumValue.GetType().GetField(enumValue.ToString());
            if (field == null)
                return enumValue.ToString();

            var attributes = field.GetCustomAttributes(typeof(DescriptionAttribute), false);
            if (Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute)) is DescriptionAttribute attribute)
            {
                return attribute.Description;
            }

            return enumValue.ToString();
        }

        public static T GetEnumFromDescription<T>(this string description) where T : Enum
        {
            foreach (var field in typeof(T).GetFields())
            {
                if (Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute)) is DescriptionAttribute attribute)
                {
                    if (attribute.Description == description)
                    {
                        return (T)field.GetValue(null);
                    }
                }
                else if (field.Name == description)
                {
                    return (T)field.GetValue(null);
                }
            }
            throw new ArgumentException($"No enum value with description '{description}' found in {typeof(T).Name}.");
        }

        public static object GetEnumFromDescription(this string description, Type enumType)
        {
            if (!enumType.IsEnum)
            {
                throw new ArgumentException("The provided type is not an enum.", nameof(enumType));
            }

            foreach (var field in enumType.GetFields())
            {
                if (Attribute.GetCustomAttribute(field, typeof(DescriptionAttribute)) is DescriptionAttribute attribute)
                {
                    if (attribute.Description == description)
                    {
                        return field.GetValue(null);
                    }
                }
                else if (field.Name == description)
                {
                    return field.GetValue(null);
                }
            }

            throw new ArgumentException($"No enum value with description '{description}' found in {enumType.Name}.");
        }

        public static ModelConfigurationViewModel CreateModelConfiguration(this TomlTable modelTable, ModelType modelType)
        {
            ModelConfigurationViewModel modelConfig = modelType switch
            {
                ModelType.Chroma => new ChromaModelConfigurationViewModel(),
                ModelType.Cosmos => new CosmosModelConfigurationViewModel(),
                ModelType.Flux => new FluxModelConfigurationViewModel(),
                ModelType.Hunyuan => new HunyuanModelConfigurationViewModel(),
                ModelType.LTX => new LtxModelConfigurationViewModel(),
                ModelType.Lumina => new LuminaModelConfigurationViewModel(),
                ModelType.SDXL => new SdxlModelConfigurationViewModel(),
                ModelType.Wan21 => new WanModelConfigurationViewModel(),
                _ => throw new NotSupportedException($"Model type '{modelType}' is not supported."),
            };

            foreach (var keyValuePair in modelTable)
            {
                var property = modelConfig.GetType().GetProperties()
                   .FirstOrDefault(p =>
                   {
                       var dataMemberAttribute = p.GetCustomAttribute<DataMemberAttribute>();
                       return dataMemberAttribute != null && dataMemberAttribute.Name == keyValuePair.Key;
                   });

                if (property != null)
                {
                    var value = keyValuePair.Value;
                    if (value is TomlTable)
                    {
                        // Handle nested objects if needed
                    }
                    else
                    {
                        if (property.PropertyType.IsEnum)
                        {
                            string enumDescription = value.ToString();
                            try
                            {
                                var enumValue = enumDescription.GetEnumFromDescription(property.PropertyType);
                                property.SetValue(modelConfig, enumValue);
                            }
                            catch (ArgumentException ex)
                            {
                                throw new InvalidOperationException($"Failed to parse enum description '{enumDescription}' for property '{property.Name}'.", ex);
                            }
                        }
                        else
                        {
                            property.SetValue(modelConfig, Convert.ChangeType(value, property.PropertyType));
                        }
                    }
                }
            }

            return modelConfig;
        }

        public static TrainConfigurationViewModel DeserializeTrainConfiguration(this string tomlContent)
        {
            var tomlTable = Toml.ToModel(tomlContent);

            var trainConfig = Toml.ToModel<TrainConfigurationViewModel>(tomlContent, null, new TomlModelOptions { IgnoreMissingProperties = true });

           

            if (tomlTable.TryGetValue("model", out var modelTableObject) && modelTableObject is TomlTable modelTable)
            {
                if (!modelTable.TryGetValue("type", out var typeValue))
                {
                    throw new InvalidOperationException("The 'type' field is missing in the model configuration.");
                }

                string typeDescription = typeValue.ToString();
                var modelType = typeDescription.GetEnumFromDescription<ModelType>();

                var modelConfiguration = CreateModelConfiguration(modelTable, modelType);

                trainConfig.ModelConfiguration = modelConfiguration;

                switch (modelType)
                {
                    case ModelType.SDXL:
                        trainConfig.SdxlModelConfiguration = (SdxlModelConfigurationViewModel)modelConfiguration;
                        break;
                    case ModelType.Flux:
                        trainConfig.FluxModelConfiguration = (FluxModelConfigurationViewModel)modelConfiguration;
                        break;
                    case ModelType.LTX:
                        trainConfig.LtxModelConfiguration = (LtxModelConfigurationViewModel)modelConfiguration;
                        break;
                    case ModelType.Hunyuan:
                        trainConfig.HunyuanModelConfiguration = (HunyuanModelConfigurationViewModel)modelConfiguration;
                        break;
                    case ModelType.Cosmos:
                        trainConfig.CosmosModelConfiguration = (CosmosModelConfigurationViewModel)modelConfiguration;
                        break;
                    case ModelType.Lumina:
                        trainConfig.LuminaModelConfiguration = (LuminaModelConfigurationViewModel)modelConfiguration;
                        break;
                    case ModelType.Wan21:
                        trainConfig.WanModelConfiguration = (WanModelConfigurationViewModel)modelConfiguration;
                        break;
                    case ModelType.Chroma:
                        trainConfig.ChromaModelConfiguration = (ChromaModelConfigurationViewModel)modelConfiguration;
                        break;
                }
            }

            return trainConfig;
        }

        public static string GenerateToml(this DatasetConfigurationViewModel viewModel)
        {
            var tomlTable = new TomlTable
            {
                { "name", viewModel.Name },
                { "model", viewModel.Model.GetDescription() },
                { "enable_ar_bucket", viewModel.EnableARBucket },
                { "min_ar", viewModel.MinAR },
                { "max_ar", viewModel.MaxAR },
                { "num_ar_buckets", viewModel.NumARBuckets },
            };

            if (!string.IsNullOrEmpty(viewModel.ResolutionsJson))
            {
                try
                {
                    var resolutions = viewModel.ResolutionsJson.ParseJsonToTomlValue();
                    if (resolutions != null)
                    {
                        tomlTable.Add("resolutions", resolutions);
                    }
                }
                catch (InvalidOperationException ex)
                {
                    Console.WriteLine($"Error processing resolutions: {ex.Message}");
                }
            }

            if (!string.IsNullOrEmpty(viewModel.FrameBucketsJson))
            {
                try
                {
                    var frameBuckets = viewModel.FrameBucketsJson.ParseJsonToTomlValue();
                    if (frameBuckets != null)
                    {
                        tomlTable.Add("frame_buckets", frameBuckets);
                    }
                }
                catch (InvalidOperationException ex)
                {
                    Console.WriteLine($"Error processing frame_buckets: {ex.Message}");
                }
            }

            if (!string.IsNullOrEmpty(viewModel.ARBucketsJson))
            {
                try
                {
                    var arBuckets = viewModel.ARBucketsJson.ParseJsonToTomlValue();
                    if (arBuckets != null)
                    {
                        tomlTable.Add("ar_buckets", arBuckets);
                    }
                }
                catch (InvalidOperationException ex)
                {
                    Console.WriteLine($"Error processing ar_buckets: {ex.Message}");
                }
            }

            foreach (var directory in viewModel.Directories!)
            {
                var directoryTable = new TomlTable
                {
                    { "path", directory.Path },
                    { "num_repeats", directory.NumRepeats },
                };

                if (!string.IsNullOrEmpty(directory.ResolutionsJson))
                {
                    try
                    {
                        var resolutions = directory.ResolutionsJson.ParseJsonToTomlValue();
                        if (resolutions != null)
                        {
                            directoryTable.Add("resolutions", resolutions);
                        }
                    }
                    catch (InvalidOperationException ex)
                    {
                        Console.WriteLine($"Error processing resolutions: {ex.Message}");
                    }
                }

                if (!string.IsNullOrEmpty(directory.FrameBucketsJson))
                {
                    try
                    {
                        var frameBuckets = directory.FrameBucketsJson.ParseJsonToTomlValue();
                        if (frameBuckets != null)
                        {
                            directoryTable.Add("frame_buckets", frameBuckets);
                        }
                    }
                    catch (InvalidOperationException ex)
                    {
                        Console.WriteLine($"Error processing frame_buckets: {ex.Message}");
                    }
                }

                if (!string.IsNullOrEmpty(directory.ARBucketsJson))
                {
                    try
                    {
                        var arBuckets = directory.ARBucketsJson.ParseJsonToTomlValue();
                        if (arBuckets != null)
                        {
                            directoryTable.Add("ar_buckets", arBuckets);
                        }
                    }
                    catch (InvalidOperationException ex)
                    {
                        Console.WriteLine($"Error processing ar_buckets: {ex.Message}");
                    }
                }

                if (!tomlTable.ContainsKey("directory"))
                {
                    tomlTable.Add("directory", new TomlTableArray());
                }

                ((TomlTableArray)tomlTable["directory"]).Add(directoryTable);
            }

            return Toml.FromModel(tomlTable);
        }

        public static string GenerateToml(this TrainConfigurationViewModel viewModel)
        {
            if (viewModel.DatasetConfig != null && viewModel.OutputDir != null)
            {

                var tomlTable = new TomlTable
                {
                    { "output_dir", viewModel.OutputDir },
                    { "dataset", viewModel.DatasetConfig },
                    { "epochs", viewModel.Epochs },
                    { "micro_batch_size_per_gpu", viewModel.MicroBatchSizePerGPU },
                    { "pipeline_stages", viewModel.PipelineStages },
                    { "gradient_accumulation_steps", viewModel.GradientAccumulationSteps },
                    { "gradient_clipping", viewModel.GradientClipping },
                    { "warmup_steps", viewModel.WarmupSteps },
                    { "eval_every_n_epochs", viewModel.EvalEveryNEpochs },
                    { "eval_before_first_step", viewModel.EvalBeforeFirstSteps },
                    { "eval_micro_batch_size_per_gpu", viewModel.EvalMicroBatchSizePerGPU },
                    { "eval_gradient_accumulation_steps", viewModel.EvalGradientAccumulationSteps },
                    { "save_every_n_epochs", viewModel.SaveEveryNEpochs },
                    { "checkpoint_every_n_minutes", viewModel.CheckpointEveryNMinutes },
                    { "activation_checkpointing", viewModel.ActivationCheckpointing },
                    { "partition_method", viewModel.PartitionMethod.GetDescription() },
                    { "save_dtype", viewModel.SaveDType.GetDescription() },
                    { "caching_batch_size", viewModel.CachingBatchSize },
                    { "steps_per_print", viewModel.StepsPerPrint },
                    { "video_clip_mode", viewModel.VideoClipMode.GetDescription() }
                };

                if (viewModel.BlocksToSwap != null)
                {
                    tomlTable.Add("blocks_to_swap", viewModel.BlocksToSwap!);
                }

                if (viewModel.ModelConfiguration != null)
                {
                    var modelTable = new TomlTable
                {
                    { "type", viewModel.ModelConfiguration.Type.GetDescription() },
                    { "dtype", viewModel.ModelConfiguration.Dtype.GetDescription() },
                };

                    var currentModelConfiguration = viewModel.GetCurrentModelConfiguration();

                    switch (viewModel.ModelConfiguration.Type)
                    {
                        case Enums.ModelType.SDXL:
                            var sdxlConfig = (SdxlModelConfigurationViewModel)currentModelConfiguration;

                            if (sdxlConfig != null)
                            {
                                modelTable.Add("checkpoint_path", sdxlConfig.CheckpointPath!);

                                if (sdxlConfig.VPred != null && sdxlConfig.VPred == true)
                                {
                                    modelTable.Add("v_pred", sdxlConfig.VPred);
                                }

                                if (sdxlConfig.MinSnrGamma != null && sdxlConfig.MinSnrGamma > 0)
                                {
                                    modelTable.Add("min_snr_gamma", sdxlConfig.MinSnrGamma);
                                }

                                if (sdxlConfig.DebiasedEstimationLoss != null && sdxlConfig.DebiasedEstimationLoss == true)
                                {
                                    modelTable.Add("debiased_estimation_loss", sdxlConfig.DebiasedEstimationLoss);
                                }

                                modelTable.Add("unet_lr", sdxlConfig.UnetLr);
                                modelTable.Add("text_encoder_1_lr", sdxlConfig.TextEncoder1Lr);
                                modelTable.Add("text_encoder_2_lr", sdxlConfig.TextEncoder2Lr);
                            }
                            break;
                        case Enums.ModelType.Flux:
                            var fluxConfig = (FluxModelConfigurationViewModel)currentModelConfiguration;
                            if (fluxConfig != null)
                            {
                                modelTable.Add("diffusers_path", fluxConfig.DiffusersPath!);

                                if (!string.IsNullOrEmpty(fluxConfig.TransformerPath))
                                {
                                    modelTable.Add("transformer_path", fluxConfig.TransformerPath);
                                }

                                if (fluxConfig.TransformerDType != null)
                                {
                                    modelTable.Add("transformer_dtype", fluxConfig.TransformerDType.GetDescription());
                                }

                                if (fluxConfig.BypassGuidanceEmbedding != null)
                                {
                                    modelTable.Add("bypass_guidance_embedding", fluxConfig.BypassGuidanceEmbedding);
                                }

                                modelTable.Add("flux_shift", fluxConfig.FluxShift);
                            }
                            break;
                        case Enums.ModelType.LTX:
                            var ltxConfig = (LtxModelConfigurationViewModel)currentModelConfiguration;
                            if (ltxConfig != null)
                            {
                                modelTable.Add("diffusers_path", ltxConfig.DiffusersPath!);
                                modelTable.Add("single_file_path", ltxConfig.SingleFilePath!);
                                modelTable.Add("timestep_sample_method", ltxConfig.TimestepSampleMethod!.GetDescription());
                            }
                            break;
                        case Enums.ModelType.Hunyuan:
                            var hunyuanConfig = (HunyuanModelConfigurationViewModel)currentModelConfiguration;
                            if (hunyuanConfig != null)
                            {
                                if (hunyuanConfig.IsOfficialCheckpoint != null && hunyuanConfig.IsOfficialCheckpoint == true)
                                {
                                    modelTable.Add("ckpt_path", hunyuanConfig.CkptPath!);
                                }
                                else
                                {
                                    modelTable.Add("transformer_path", hunyuanConfig.TransformerPath!);
                                    modelTable.Add("transformer_dtype", hunyuanConfig.TransformerDType!.GetDescription());
                                    modelTable.Add("vae_path", hunyuanConfig.VaePath!);
                                    modelTable.Add("llm_path", hunyuanConfig.LlmPath!);
                                    modelTable.Add("clip_path", hunyuanConfig.ClipPath!);

                                }


                                modelTable.Add("timestep_sample_method", hunyuanConfig.TimestepSampleMethod!.GetDescription());
                            }
                            break;
                        case Enums.ModelType.Cosmos:
                            var cosmosConfig = (CosmosModelConfigurationViewModel)currentModelConfiguration;
                            if (cosmosConfig != null)
                            {
                                modelTable.Add("transformer_path", cosmosConfig.TransformerPath!);
                                modelTable.Add("vae_path", cosmosConfig.VaePath!);
                                modelTable.Add("text_encoder_path", cosmosConfig.TextEncoderPath!);
                            }
                            break;
                        case Enums.ModelType.Lumina:
                            var luminaConfig = (LuminaModelConfigurationViewModel)currentModelConfiguration;
                            if (luminaConfig != null)
                            {
                                modelTable.Add("transformer_path", luminaConfig.TransformerPath!);
                                modelTable.Add("llm_path", luminaConfig.LlmPath!);
                                modelTable.Add("vae_path", luminaConfig.VaePath!);
                                modelTable.Add("lumina_shift", luminaConfig.LuminaShift!);
                            }
                            break;
                        case Enums.ModelType.Wan21:
                            var wanConfig = (WanModelConfigurationViewModel)currentModelConfiguration;
                            if (wanConfig != null)
                            {
                                modelTable.Add("ckpt_path", wanConfig.CkptPath!);
                                if (wanConfig.TransformerDType != null)
                                {
                                    modelTable.Add("transformer_dtype", wanConfig.TransformerDType!.GetDescription());
                                }
                                modelTable.Add("timestep_sample_method", wanConfig.TimestepSampleMethod!.GetDescription());
                            }
                            break;
                        case Enums.ModelType.Chroma:
                            var chromaConfig = (ChromaModelConfigurationViewModel)currentModelConfiguration;
                            if (chromaConfig != null)
                            {
                                modelTable.Add("diffusers_path", chromaConfig.DiffusersPath!);
                                modelTable.Add("transformer_path", chromaConfig.TransformerPath!);
                                modelTable.Add("transformer_dtype", chromaConfig.TransformerDType!.GetDescription());
                                modelTable.Add("flux_shift", chromaConfig.FluxShift!);
                            }
                            break;
                    }

                    tomlTable.Add("model", modelTable);
                }

                if (viewModel.AdapterConfiguration != null)
                {
                    var adapterTable = new TomlTable
                    {
                        { "type", viewModel.AdapterConfiguration.Type },
                        { "rank", viewModel.AdapterConfiguration.Rank },
                        { "dtype", viewModel.AdapterConfiguration.Dtype.GetDescription() }
                    };

                    if (!string.IsNullOrEmpty(viewModel.AdapterConfiguration.InitFromExisting))
                    {
                        adapterTable.Add("init_from_existing", viewModel.AdapterConfiguration.InitFromExisting!);
                    }

                    tomlTable.Add("adapter", adapterTable);
                }


                if (viewModel.OptimizerConfiguration != null)
                {
                    var optimizerTable = new TomlTable
                {
                    { "type", viewModel.OptimizerConfiguration.Type.GetDescription() },
                    { "lr", viewModel.OptimizerConfiguration.Lr },
                    { "weight_decay", viewModel.OptimizerConfiguration.WeightDecay },
                    { "eps", viewModel.OptimizerConfiguration.Eps }
                };

                    if (viewModel.OptimizerConfiguration.Betas != null)
                    {
                        optimizerTable.Add("betas", viewModel.OptimizerConfiguration.Betas);
                    }

                    tomlTable.Add("optimizer", optimizerTable);
                }

                return Toml.FromModel(tomlTable);
            }
            else
            {
                throw new InvalidOperationException("The dataset configuration or output directory is missing.");
            }
        }

        public static void EnsureDirectoriesExist(this AppSettingsConfiguration settings)
        {
            FileHelper.CreateDirectoryIfNotExists(settings.DatasetsPath);
            FileHelper.CreateDirectoryIfNotExists(settings.ConfigsPath);
            FileHelper.CreateDirectoryIfNotExists(settings.OutputsPath);
            FileHelper.CreateDirectoryIfNotExists(settings.ModelsPath);
        }

        public static void EnsureDirectoriesExist(this AppSettingsConfiguration settings, string datasetname)
        {
            FileHelper.CreateDirectoryIfNotExists(Path.Combine(settings.DatasetsPath, datasetname));
            FileHelper.CreateDirectoryIfNotExists(Path.Combine(settings.ConfigsPath, datasetname));
            FileHelper.CreateDirectoryIfNotExists(Path.Combine(settings.OutputsPath, datasetname));
        }

        public static string ConvertTomlValueToJson(this object tomlValue)
        {
            if (tomlValue is TomlArray tomlArray)
            {
                if (tomlArray.Count > 0 && tomlArray[0] is TomlArray)
                {
                    var nestedList = tomlArray
                        .Select(innerArray => ((TomlArray)innerArray)
                            .Select(item => Convert.ToInt32(item))
                            .ToList()) 
                        .ToList();
                    return JsonSerializer.Serialize(nestedList);
                }
                else
                {
                    var simpleArray = tomlArray
                        .Select(item => Convert.ToInt32(item))
                        .ToArray();
                    return JsonSerializer.Serialize(simpleArray);
                }
            }

            return string.Empty;
        }

        public static object? ParseJsonToTomlValue(this string json)
        {
            if (string.IsNullOrEmpty(json))
            {
                return null;
            }

            try
            {
                var array = JsonSerializer.Deserialize<int[]>(json);
                if (array != null)
                {
                    return array; // Returns the simple array
                }
            }
            catch (JsonException)
            {
                var matrix = JsonSerializer.Deserialize<int[][]>(json);
                if (matrix != null)
                {
                    return matrix; // Returns the matrix
                }
            }

            throw new InvalidOperationException("The JSON is neither a simple array nor a matrix.");
        }

        public static void ResetObjectToDefault<T>(T obj) where T : class, new()
        {
            if (obj == null) return;

            var defaultInstance = new T();

            foreach (var property in typeof(T).GetProperties())
            {
                if (!property.CanWrite) continue;

                var defaultValue = property.GetValue(defaultInstance);
                property.SetValue(obj, defaultValue);
            }
        }

        public static void CopyProperties(this TrainConfigurationViewModel source, TrainConfigurationViewModel target)
        {
            foreach (var prop in typeof(ModelConfigurationViewModel).GetProperties())
            {
                if (prop.CanWrite)
                {
                    prop.SetValue(target, prop.GetValue(source));
                }
            }
        }

        private static string FormatFloat(float value)
        {
            if (Math.Abs(value) < 0.001f)
                return value.ToString("0.#####E+0", CultureInfo.InvariantCulture);
            else
                return value.ToString("0.#####", CultureInfo.InvariantCulture);
        }

    }
}
