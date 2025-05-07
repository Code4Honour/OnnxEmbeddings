using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxEmbeddings.Models;
using OnnxEmbeddings.Tokenizers;

namespace OnnxEmbeddings.Transformers
{

    /// <summary>
    /// Not Finished
    /// </summary>
    public class Medembed
    {
        private Medembed()
        {

        }
        private static object locker = new object();
        private static Medembed? _Instance;
        public static Medembed Instance
        {
            get
            {
                lock (locker)
                {
                    if (_Instance == null)
                    {
                        _Instance = new Medembed();
                    }
                }
                return _Instance;
            }
        }

        public List<(string, float[])> Encode(string[] sentences, EmbeddingModels embeddingType)
        {
            if (embeddingType != EmbeddingModels.medembed_small)
            {
                throw new Exception("Invalid embedding.  Use medembed embedding");
            }
            string? onnxPath = Util.GetModelName(embeddingType);
            sentences = sentences.Where(x => x != null && x != "").ToArray();
            TokenizationResult inputs = default;
            inputs = TokenizerBGE.Instance.Tokenize(sentences);

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
            output = session.Run(inputContainer);

            //Console.WriteLine("Inference Complete");
            //Console.WriteLine($"Compute Time {dif.ToString("N0")} Milliseconds");
            Tensor<float>? onnxOutput = output.FirstOrDefault()?.AsTensor<float>();
       
            return null;
        }
    }
}
