using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxEmbeddings.Models;
using OnnxEmbeddings.Tokenizers;
using TorchSharp;
using F = TorchSharp.torch.nn.functional;
using TorchTensor = TorchSharp.torch.Tensor;

namespace OnnxEmbeddings.Transformers
{
    public class Mxbai
    {
        private Mxbai()
        {
        }
        private static object locker = new object();
        private static Mxbai? _Instance;
        public static Mxbai Instance
        {
            get
            {
                lock (locker)
                {
                    if (_Instance == null)
                    {
                        _Instance = new Mxbai();
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
            if (embeddingType != EmbeddingModels.mxbai_embed_large_v1)
            {
                throw new Exception("Invalid embedding.  Use mxabi embedding");
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
            Tensor<float>? model_output = session.Run(inputContainer).FirstOrDefault()?.AsTensor<float>();

            if (model_output == null) { throw new Exception("Invalid - ONNX output was null"); }

            TorchTensor embeddings = pool(model_output, attentionMaskTensor, "mean");

            List<(string, float[])> embeddingsDictionary = new List<(string, float[])>();
            int count = model_output.Dimensions[0];

            for (int i = 0; i < count; i++)
            {
                float[] embeddingData = embeddings[i].data<float>().ToArray();
                string sentence = sentences[i];
                embeddingsDictionary.Add((sentence, embeddingData));
            }

            DateTime d2 = DateTime.UtcNow; double dif = (d2 - d1).TotalMilliseconds;
            Console.WriteLine($"Inference Complete - Compute Time {dif.ToString("N0")} Milliseconds");
            return embeddingsDictionary;
        }
       
        public List<(string, float[])> EncodeLive(string[] sentences, EmbeddingModels embeddingType)
        {
            DateTime d1 = DateTime.UtcNow;

            if (embeddingType != EmbeddingModels.mxbai_embed_large_v1 && embeddingType != EmbeddingModels.mxbai_embed_xsmall_v1)
            {
                throw new Exception("Invalid embedding.  Use mxabi embedding");
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

            List<NamedOnnxValue> inputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor),
             
            };
            if(embeddingType == EmbeddingModels.mxbai_embed_large_v1)
            {
                DenseTensor<long> tokenTypeIdsTensor = new DenseTensor<long>(inputs.token_type_ids_flat(), dims);
                NamedOnnxValue namedValue = NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor);
                inputContainer.Add(namedValue);
            }
            Tensor<float>? model_output = sessionLive.Run(inputContainer).FirstOrDefault()?.AsTensor<float>();

            if (model_output == null) { throw new Exception("Invalid - ONNX output was null"); }

            TorchTensor embeddings = pool(model_output, attentionMaskTensor, "mean");

            List<(string, float[])> embeddingsDictionary = new List<(string, float[])>();
            int count = model_output.Dimensions[0];

            for (int i = 0; i < count; i++)
            {
                float[] embeddingData = embeddings[i].data<float>().ToArray();
                string sentence = sentences[i];
                embeddingsDictionary.Add((sentence, embeddingData));
            }

            DateTime d2 = DateTime.UtcNow; double dif = (d2 - d1).TotalMilliseconds;
            Console.WriteLine($"Inference Complete - Compute Time {dif.ToString("N0")} Milliseconds");
            return embeddingsDictionary;
        }

        public List<(string, float[])> EncodeString(TokenizationResult inputs, string[] sentences, EmbeddingModels embeddingType)
        {
            DateTime d1 = DateTime.UtcNow;

            if (embeddingType != EmbeddingModels.mxbai_embed_large_v1 && embeddingType != EmbeddingModels.mxbai_embed_xsmall_v1)
            {
                throw new Exception("Invalid embedding.  Use mxabi embedding");
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

            List<NamedOnnxValue> inputContainer = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
                NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
            };
            if (embeddingType == EmbeddingModels.mxbai_embed_large_v1)
            {
                DenseTensor<long> tokenTypeIdsTensor = new DenseTensor<long>(inputs.token_type_ids_flat(), dims);
                inputContainer.Add(NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIdsTensor));
            }
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
      
        public TorchTensor pool(Tensor<float> output, DenseTensor<long> inputs, string strategy = "cls")
        {
            if (strategy == "cls")
            {
                TorchTensor tensor = Util.ToTorchTensor(output, device);
                tensor = tensor[torch.TensorIndex.Colon, 0];
                return tensor;
            }
            else if (strategy == "mean")
            {
                TorchTensor tensor = Util.ToTorchTensor(output, device);
                TorchTensor mask = Util.ToTorchTensor(inputs, device);
                tensor = Util.MeanPooling(tensor, mask);
                return tensor;
            }
            else
            {
                throw new NotImplementedException("Error. Please use 'cls' (Classification Token) or 'mean' (Mean Pooling).");
            }
        }
    }
}
