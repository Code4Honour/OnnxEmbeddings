using ICSharpCode.SharpZipLib.Core;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxEmbeddings.Models;
using System;
using System.Net;
using System.Net.Http.Json;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using TorchSharp;
using static System.Net.WebRequestMethods;
using static TorchSharp.torch;
using TorchTensor = TorchSharp.torch.Tensor;

namespace OnnxEmbeddings
{
    public class Util
    {
        public static string Drive { get; set; } =  "F:";
        public static string EmbeddingFolder { get; set; } = "\\OnnxEmbeddings";
        public static InferenceSession MakeSession(string onnxPath)
        {
            SessionOptions gpu = SessionOptions.MakeSessionOptionWithCudaProvider(0);
            InferenceSession session = new InferenceSession(onnxPath, gpu);
            return session;
        }

        public static string? GetModelName<T>(T modelId) where T : Enum
        {
            FieldInfo? field = typeof(T).GetField(modelId.ToString());
            EnumFlagAttribute? attribute = field?.GetCustomAttribute<EnumFlagAttribute>();
            if (attribute != null)
            {
                return attribute.Name;
            }
            throw new ArgumentException($"Invalid model ID: {modelId}");
        }

        public static (string modelName, string folder) GetModelNameAndFolder<T>(T modelId) where T : Enum
        {
            
            FieldInfo? field = typeof(T).GetField(modelId.ToString());
            EnumFlagAttribute? attribute = field?.GetCustomAttribute<EnumFlagAttribute>();
            if (attribute != null)
            {
                string file = Path.Join(Drive + EmbeddingFolder, attribute.Name);
                FileInfo info = new FileInfo(file);
                if(info.Exists == false)
                {
                    DownloadFile(modelId);
                }

                return (file, info.DirectoryName);
            }
            throw new ArgumentException($"Invalid model ID: {modelId}");
        }

        private static  void DownloadFile<T>(T modelId) where T : Enum
        {
            if (typeof(T) != (typeof(EmbeddingModels)))
            {
                throw new Exception("Enum Mismatch");
            }
            string downloadString = default;
            string downloadStringAux = default;
            switch (modelId)
            {
                case EmbeddingModels.bge_m3:
                    downloadString = "https://huggingface.co/BAAI/bge-m3/blob/main/onnx/model.onnx";
                    downloadStringAux = "https://huggingface.co/BAAI/bge-m3/blob/main/onnx/model.onnx_data";
                    break;
                case EmbeddingModels.bge_large_en_v1_5:
                    downloadString = "https://huggingface.co/BAAI/bge-large-en-v1.5/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.bge_base_en_v1_5:
                    downloadString = "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx?download=true";
                    break;
                case EmbeddingModels.jina_embeddings_v3_570M:
                    downloadString = "https://huggingface.co/jinaai/jina-embeddings-v3/blob/main/onnx/model.onnx";
                    downloadStringAux = "https://huggingface.co/jinaai/jina-embeddings-v3/blob/main/onnx/model.onnx_data";
                    break;
                case EmbeddingModels.jina_embeddings_v2_base_en_137M:
                    downloadString = "https://huggingface.co/jinaai/jina-embeddings-v2-base-en/blob/main/model.onnx";
                    break;
                case EmbeddingModels.jina_embeddings_v2_small_en_33M:
                    downloadString = "https://huggingface.co/jinaai/jina-embeddings-v2-small-en/blob/main/model.onnx";
                    break;
                case EmbeddingModels.jina_embeddings_v2_base_code_161M:
                    downloadString = "https://huggingface.co/jinaai/jina-embeddings-v2-base-code/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.all_MiniLM_L12_v2:
                    downloadString = "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.all_MiniLM_L6_v2:
                    downloadString = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.all_mpnet_base_v2:
                    downloadString = "https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.mxbai_embed_large_v1:
                    downloadString = "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.mxbai_embed_xsmall_v1:
                    downloadString = "https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.nomic_embed_text_v1_5:
                    downloadString = "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.snowflake_arctic_embed_xs_22M:
                    downloadString = "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.snowflake_arctic_embed_s_33M:
                    downloadString = "https://huggingface.co/Snowflake/snowflake-arctic-embed-s/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.snowflake_arctic_embed_m_110M:
                    downloadString = "https://huggingface.co/Snowflake/snowflake-arctic-embed-m/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.snowflake_arctic_embed_l_335M:
                    downloadString = "https://huggingface.co/Snowflake/snowflake-arctic-embed-l/blob/main/onnx/model.onnx";
                    break;
                case EmbeddingModels.medembed_small:
                    break;
                case EmbeddingModels.S_PubMedBert_MS_MARCO:
                    break;
                case EmbeddingModels.pubmedbert_base_embeddings:
                    break;
                case EmbeddingModels.legal_roberta_base:
                    break;
                case EmbeddingModels.legal_roberta_base_uncase:
                    break;
                case EmbeddingModels.codet5p_110m_embedding:
                    break;
                case EmbeddingModels.codebert_base:
                    break;
                default:
                    break;
            }

            try
            {
                if (downloadString != default)
                {
                    Console.WriteLine("Downloading model. Please Wait");
                    string path = modelId.GetEnumFlagValue();
                    if (path == null)
                    {
                        Console.WriteLine("Invalid Model");
                        return;
                    }
                    string saveLocation = Path.Join(Drive + EmbeddingFolder, path);
                    DownloadFile(downloadString, saveLocation);
                }

                if (downloadStringAux != default)
                {
                    Console.WriteLine("Downloading additional model files. Please Wait");
                    string pathAux = modelId.GetEnumFlagValueAux();
                    if (pathAux == null)
                    {
                        Console.WriteLine("Invalid Model");
                        return;
                    }
                    string saveLocationAux = Path.Join(Drive + EmbeddingFolder, pathAux);
                    DownloadFile(downloadStringAux, saveLocationAux);
                }


            }
            catch (Exception e)
            {
                Console.WriteLine(e.Message);
            }
        }

        private static void DownloadFile(string downloadString, string saveLocation)
        {
            using (HttpClient client = new HttpClient())
            {

                using (HttpResponseMessage response = client.GetAsync(downloadString, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult())
                {
                    response.EnsureSuccessStatusCode();

                    using (Stream stream = response.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
                    {
                        using (FileStream fileStream = new FileStream(saveLocation, FileMode.Create, FileAccess.Write, FileShare.None))
                        {
                            stream.CopyToAsync(fileStream).GetAwaiter().GetResult();
                        }
                    }
                }
            }
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string ReplaceLigatures(string input)
        {
            return input
                .Replace("\uFB00", "ff")   // ﬀ
                .Replace("\uFB01", "fi")   // ﬁ
                .Replace("\uFB02", "fl")   // ﬂ
                .Replace("\uFB03", "ffi")  // ﬃ
                .Replace("\uFB04", "ffl")  // ﬄ
                .Replace("\uFB05", "ft")   // ﬅ
                .Replace("\uFB06", "st")  // ﬆ
                .Replace("\ud83c", "s")
                .Replace("\uddfa", "u")
                .Replace("\uddf8", "s")      // Emoji Symbols
                //.Replace("\U0001F947", "[1st Place Medal]")         // 🥇
                //.Replace("\U0001F56F", "[Candle]")                  // 🕯️
                //.Replace("\U0001F521", "[Input Symbols]")           // 🔡
                //.Replace("\U0001F3E5", "[Hospital]")                // 🏥
                //.Replace("\U00002695", "[Medical Symbol]")          // ⚕️
                //.Replace("\U0001F4C4", "[Document]")                // 📄
                //.Replace("\U000026A1", "[Lightning Bolt]")          // ⚡
                //.Replace("\U0001F4A1", "[Light Bulb]")              // 💡
                //.Replace("\U0001F316", "[Waning Gibbous Moon]")     // 🌖
                //.Replace("\U0001F916", "[Robot]")                   // 🤖
                //.Replace("\U0001F525", "[Fire]")                    // 🔥
                //.Replace("\U00002696", "[Scales]")                  // ⚖️
                //.Replace("\U0001F56F\uFE0F", "[Candle]")            // 🕯️ (with variation selector)
                //.Replace("\U0001F4A1", "[Idea]")
                ;                  
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string FullClean(string sentence)
        {
            sentence = sentence.Replace("\r", "").Replace("\n", " ").Replace("\t", " ");
            sentence = AddSpaceAroundNonAlphanumeric(sentence);
            sentence = ReplaceLigatures(sentence);
            string item = sentence.ToLower();
            return item;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string PartialClean(string sentence)
        {
            sentence = sentence.Replace("\r", "").Replace("\n", " ").Replace("\t", " ");
            sentence = AddSpaceAroundNonAlphanumeric(sentence);
            sentence = sentence.ToLower();
            return sentence;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string AddSpaceAroundNonAlphanumeric(string input)
        {
            input =  Regex.Replace(input, @"(\W+)", " $1 ");
            input = Regex.Replace(input, @"[^a-zA-Z0-9\s]", " $0 ").Replace("  ", " ").Trim();
            return input;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static string ReplaceFirstMatch(string input, string oldValue, string newValue)
        {
            int index = input.IndexOf(oldValue);
            if (index >= 0)
            {
                return input.Substring(0, index) + newValue + input.Substring(index + oldValue.Length);
            }
            return input;
        }

        public static TorchTensor ToTorchTensor(Tensor<float> model_output, Device device)
        {
            float[] r = model_output.ToArray();
            long[] tensorDims = model_output.Dimensions.ToArray().Select(x => (long)x).ToArray();
            TorchTensor t_output = torch.tensor(r, tensorDims, device: device);
            return t_output;
        }

        public static TorchTensor ToTorchTensor(Tensor<long> model_output, Device device)
        {
            long[] r = model_output.ToArray();
            long[] tensorDims = model_output.Dimensions.ToArray().Select(x => (long)x).ToArray();
            TorchTensor t_output = torch.tensor(r, tensorDims, device: device);
            return t_output;
        }

        public static TorchTensor ToTorchTensor(long[][] data, Device device)
        {
            int outerLength = data.Length;
            int innerLength = data[0].Length; 
            long[] flatData = new long[outerLength * innerLength];
            for (int i = 0; i < outerLength; i++)
            {
                if (data[i].Length != innerLength)
                {
                    throw new ArgumentException("All sub-arrays must have the same length.");
                }
                Array.Copy(data[i], 0, flatData, i * innerLength, innerLength);
            }
            return torch.tensor(flatData, new long[] { outerLength, innerLength }, dtype: ScalarType.Int64, device: device);
        }

        public static TorchTensor MeanPooling(TorchTensor embeddings, TorchTensor mask)
        {
            TorchTensor maskExpanded = mask.unsqueeze(2).expand_as(embeddings);
            if (maskExpanded.dtype != ScalarType.Float32)
            {
                maskExpanded = maskExpanded.to(ScalarType.Float32);
            }
            TorchTensor result = embeddings * maskExpanded;
            TorchTensor summedResult = result.sum(1);
            TorchTensor summedMask = maskExpanded.sum(1);
            TorchTensor clampedSum = torch.maximum(summedMask, torch.tensor(1e-9));
            TorchTensor final = summedResult / clampedSum;
            if (final.dtype != ScalarType.Float32)
            {
                final = final.to(ScalarType.Float32);
            }
            return final;
        }
    }
}
