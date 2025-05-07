using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxEmbeddings.Models;
using OnnxEmbeddings.Tokenizers;

namespace OnnxEmbeddings.Transformers
{

    /// <summary>
    /// Not Finished
    /// </summary>
    public class LegalRoberta
    {
        private LegalRoberta()
        {

        }
        private static object locker = new object();
        private static LegalRoberta? _Instance;
        public static LegalRoberta Instance
        {
            get
            {
                lock (locker)
                {
                    if (_Instance == null)
                    {
                        _Instance = new LegalRoberta();
                    }
                }
                return _Instance;
            }
        }

        public List<(string, float[])> Encode(string[] sentences, EmbeddingModels embeddingType)
        {
            if (embeddingType != EmbeddingModels.legal_roberta_base)
            {
                throw new Exception("Invalid embedding.  Use All-Mini embedding");
            }
            (string onnxPath, string dir) = Util.GetModelNameAndFolder(embeddingType);
            sentences = sentences.Where(x => x != null && x != "").ToArray();
            for (int i = 0; i < sentences.Length; i++)
            {
                sentences[i] = sentences[i].Replace("<mask>", "[MASK]");
            }
            TokenizationResult inputs = default;
            TokenizerBert tokenizer = new TokenizerBert(Path.Combine(dir, "tokenizer.json"), true);
            inputs = tokenizer.Tokenize(sentences, 1, 2, "Ġ");

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
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> output = default;
            Console.WriteLine("Starting Inference");
            DateTime d1 = DateTime.UtcNow;
            output = session.Run(inputContainer);
            DateTime d2 = DateTime.UtcNow;
            double dif = (d2 - d1).TotalMilliseconds;
            Console.WriteLine("Inference Complete");
            Console.WriteLine($"Compute Time {dif.ToString("N0")} Milliseconds");

            Tensor<float>? onnxOutput = output.FirstOrDefault()?.AsTensor<float>();

            throw new Exception("Completed following code later.  Normalization needs to be done.");
        }
    }
}
