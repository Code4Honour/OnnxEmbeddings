using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxEmbeddings.Models;
using OnnxEmbeddings.Tokenizers;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;
using TorchTensor = TorchSharp.torch.Tensor;

namespace OnnxEmbeddings.Transformers
{
    public class Nomic
    {
        private Nomic()
        {
        }
        private static object locker = new object();
        private static Nomic? _Instance;
        public static Nomic Instance
        {
            get
            {
                lock (locker)
                {
                    if (_Instance == null)
                    {
                        _Instance = new Nomic();
                    }
                }
                return _Instance;
            }
        }
        TokenizerBert tokenizerLive { get; set; }
        InferenceSession sessionLive { get; set; }
        torch.Device device { get; set; }
        
        public List<(string, float[])> Encode(string[] sentences, EmbeddingModels embeddingType)
        {
            if (embeddingType != EmbeddingModels.nomic_embed_text_v1_5)
            {
                throw new Exception("Invalid embedding.  Use All-Mini embedding");
            }
            (string onnxPath, string folder) = Util.GetModelNameAndFolder(embeddingType);
            sentences = sentences.Where(x => x != null && x != "").ToArray();
            var tokenizer = new TokenizerBert(Path.Combine(folder, "vocab.txt"));
            TokenizationResult inputs = tokenizer.Tokenize(sentences, 101, 102);

            Console.WriteLine("Initializing ONNX Inference Session");
            Console.WriteLine(onnxPath);
            using InferenceSession session = Util.MakeSession(onnxPath);
            Console.WriteLine("Session Loaded");
            int[] dims = [inputs.input_ids.Length, inputs.input_ids[0].Length];
            DenseTensor<long> inputIdsTensor = new DenseTensor<long>(inputs.input_ids_flat(), dims);
            DenseTensor<long> attentionMaskTensor = new DenseTensor<long>(inputs.attention_mask_flat(), dims);
            DenseTensor<long> tokenTypeIdsTensor = new DenseTensor<long>(inputs.token_type_ids_flat(), dims);

            List<NamedOnnxValue> inputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
            };
            Console.WriteLine("Starting Inference");
            DateTime d1 = DateTime.UtcNow;
            Tensor<float> ? model_output = session.Run(inputContainer).FirstOrDefault()?.AsTensor<float>();
            
            if (model_output == null) { throw new Exception("Invalid - ONNX output was null"); }

            TorchTensor t_output = Util.ToTorchTensor(model_output, device);
            TorchTensor t_mask = Util.ToTorchTensor(inputs.attention_mask, device);
            TorchTensor sentence_embeddings = Util.MeanPooling(t_output, t_mask);
            sentence_embeddings = F.normalize(sentence_embeddings, p: 2, dim: 1);
            List<(string, float[])> embeddings = new List<(string, float[])>();

            for (int i = 0; i < sentence_embeddings.shape[0]; i++)
            {
                float[] embedding = sentence_embeddings[i].data<float>().ToArray();
                string sentence = sentences[i];
                embeddings.Add((sentence, embedding));
            }
            t_output.Dispose();
            t_mask.Dispose();
            DateTime d2 = DateTime.UtcNow; double dif = (d2 - d1).TotalMilliseconds;
            Console.WriteLine($"Inference Complete - Compute Time {dif.ToString("N0")} Milliseconds");
            return embeddings;
        }
        
        public List<(string, float[])> EncodeLive(string[] sentences, EmbeddingModels embeddingType)
        {
            DateTime d1 = DateTime.UtcNow;

            if (embeddingType != EmbeddingModels.nomic_embed_text_v1_5)
            {
                throw new Exception("Invalid embedding.  Use All-Mini embedding");
            }
            (string onnxPath, string folder) = Util.GetModelNameAndFolder(embeddingType);
            sentences = sentences.Where(x => x != null && x != "").ToArray();
            if (tokenizerLive == null)
            {
                tokenizerLive = new TokenizerBert(Path.Combine(folder, "vocab.txt"));
                device = torch.device(-1);
                torch.set_grad_enabled(false);
            }
            TokenizationResult inputs = tokenizerLive.Tokenize(sentences, 101, 102);

            if (sessionLive == null)
            {
                Console.WriteLine("Initializing ONNX Inference Session");
                Console.WriteLine(onnxPath);
                sessionLive = Util.MakeSession(onnxPath);
                Console.WriteLine("Session Loaded");
            }
            int[] dims = [inputs.input_ids.Length, inputs.input_ids[0].Length];
            DenseTensor<long> inputIdsTensor = new DenseTensor<long>(inputs.input_ids_flat(), dims);
            DenseTensor<long> attentionMaskTensor = new DenseTensor<long>(inputs.attention_mask_flat(), dims);
            DenseTensor<long> tokenTypeIdsTensor = new DenseTensor<long>(inputs.token_type_ids_flat(), dims);

            List<NamedOnnxValue> inputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
            };

            Tensor<float>? model_output = sessionLive.Run(inputContainer).FirstOrDefault()?.AsTensor<float>();

            if (model_output == null) { throw new Exception("Invalid - ONNX output was null"); }

            TorchTensor t_output = Util.ToTorchTensor(model_output, device);
            TorchTensor t_mask = Util.ToTorchTensor(inputs.attention_mask, device);
            TorchTensor sentence_embeddings = Util.MeanPooling(t_output, t_mask);
            sentence_embeddings = F.normalize(sentence_embeddings, p: 2, dim: 1);
            List<(string, float[])> embeddings = new List<(string, float[])>();

            for (int i = 0; i < sentence_embeddings.shape[0]; i++)
            {
                float[] embedding = sentence_embeddings[i].data<float>().ToArray();
                string sentence = sentences[i];
                embeddings.Add((sentence, embedding));
            }
            t_output.Dispose();
            t_mask.Dispose();
            DateTime d2 = DateTime.UtcNow; double dif = (d2 - d1).TotalMilliseconds;
            Console.WriteLine($"Inference Complete - Compute Time {dif.ToString("N0")} Milliseconds");
            return embeddings;
        }
        
        public List<(string, float[])> EncodeString(TokenizationResult inputs, string[] sentences, EmbeddingModels embeddingType)
        {
            DateTime d1 = DateTime.UtcNow;

            if (embeddingType != EmbeddingModels.bge_m3 &&
                embeddingType != EmbeddingModels.bge_base_en_v1_5 &&
                embeddingType != EmbeddingModels.bge_large_en_v1_5)
            {
                throw new Exception("Invalid embedding.  Use bge embedding");
            }
            (string onnxPath, string folder) = Util.GetModelNameAndFolder(embeddingType);
            sentences = sentences.Where(x => x != null && x != "").ToArray();
            if (sessionLive == null)
            {
                Console.WriteLine("Initializing ONNX Inference Session");
                Console.WriteLine(onnxPath);
                sessionLive = Util.MakeSession(onnxPath);
                Console.WriteLine("Session Loaded");
            }
            int[] dims = [inputs.input_ids.Length, inputs.input_ids[0].Length];
            DenseTensor<long> inputIdsTensor = new DenseTensor<long>(inputs.input_ids_flat(), dims);
            DenseTensor<long> attentionMaskTensor = new DenseTensor<long>(inputs.attention_mask_flat(), dims);
            DenseTensor<long> tokenTypeIdsTensor = new DenseTensor<long>(inputs.token_type_ids_flat(), dims);
            List<NamedOnnxValue> inputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
                NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor)
            };
         
            Tensor<float>? model_output = sessionLive.Run(inputContainer)?.FirstOrDefault()?.AsTensor<float>();

            if (model_output == null) { throw new Exception("Invalid - ONNX output was null"); }

            TorchTensor sentence_embeddings = Util.ToTorchTensor(model_output, device)[torch.TensorIndex.Colon, 0];
            sentence_embeddings = F.normalize(sentence_embeddings, p: 2, dim: 1);

            List<(string, float[])> embeddings = new List<(string, float[])>();
            int count = model_output.Dimensions[0];
            for (int i = 0; i < count; i++)
            {
                float[] first = sentence_embeddings[i].data<float>().ToArray();
                string sentence = sentences[i];
                embeddings.Add((sentence, first));
            }

            sentence_embeddings.Dispose();
            DateTime d2 = DateTime.UtcNow; double dif = (d2 - d1).TotalMilliseconds;
            Console.WriteLine($"Inference Complete - Compute Time {dif.ToString("N0")} Milliseconds");
            return embeddings;
        }
    }
}
