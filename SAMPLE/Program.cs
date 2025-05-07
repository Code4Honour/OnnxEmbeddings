using OnnxEmbeddings.Models;
using OnnxEmbeddings.Tokenizers;
using OnnxEmbeddings.Transformers;
namespace Sample
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Make sure the set Util.Drive and Util.EmbeddingFolder to where the models should be saved
            // Reading the text file
            string wordFile = File.ReadAllText("sh.txt");
            string[] words = wordFile.Split(" ", StringSplitOptions.RemoveEmptyEntries);
            // Starting the tokenizer and torch (while trying to use the GPU)
            TokenizerBert tokenizer = BGE.Instance.Start(EmbeddingModels.bge_base_en_v1_5);
            // As some words will be split we cannot know the index just from the base words.  See we use 
            // an algorithm to get split the words properly with the tokenizers AND! retreive their index at the same time
            List<TokenPair> tokens = tokenizer.GetRealIndexes(wordFile);
            List<string> totalStrings = new List<string>();
            // 510 to account for prepending and appending start and finish tokens to bring the count to 512
            int sentenceLenth = 510;

            for (int i = 0; i < tokens.Count; i += sentenceLenth)
            {
                int count = Math.Min(sentenceLenth, tokens.Count - i);
                List<string> strings = tokens.GetRange(i, count).Select(t => t.Token).ToList();
                totalStrings.Add(string.Join(" ", strings));
            }
            // Processing for ONNX engine inputs
            TokenizationResult tokenResults = tokenizer.TokenizeAlreadyProcessed(totalStrings.ToArray());
            // Processing sentences 1 at a time but batches can work depending on how much GPU memory you have.
            for (int i = 0; i < totalStrings.Count; i++)
            {
                TokenizationResult chunk = tokenResults.Chunk(i, 1);
                // ONNX inference 
                List<(string, float[])> res = BGE.Instance.EncodeString(chunk, new string[] { totalStrings[i] }, EmbeddingModels.bge_base_en_v1_5);
            }
        }
    }
}
