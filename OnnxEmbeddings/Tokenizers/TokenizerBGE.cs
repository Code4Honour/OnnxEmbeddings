using System.Text.Json;
namespace OnnxEmbeddings.Tokenizers
{
    public class TokenizerBGE
    {
        private TokenizerBGE()
        {
            LoadTokenizer("bge-vocab.json");
        }
        private void LoadTokenizer(string path)
        {
            string jsonString = File.ReadAllText(path);
            Tokens = JsonSerializer.Deserialize<Dictionary<string, int>>(jsonString);
        }
        private static object locker = new object();
        private static TokenizerBGE? _Instance;
        public static TokenizerBGE Instance
        {
            get
            {
                lock (locker)
                {
                    if (_Instance == null)
                    {
                        _Instance = new TokenizerBGE();
                    }
                }
                return _Instance;
            }
        }

        Dictionary<string, int>? Tokens { get; set; }

        public TokenizationResult Tokenize(string[] sentences, string vocabPath = null)
        {
            if (vocabPath != null)
            {
                LoadTokenizer(vocabPath);
            }
            // Replace this with actual tokenization logic
            TokenizationResult tokenResult = new TokenizationResult();
            tokenResult.input_ids = new long[sentences.Length][];
            tokenResult.attention_mask = new long[sentences.Length][];
            tokenResult.token_type_ids = new long[sentences.Length][];
            for (int s = 0; s < sentences.Length; s++)
            {
                string sentence = sentences[s];
                string item = sentence.ToLower();
                string[] splits = item.Split(' ', StringSplitOptions.RemoveEmptyEntries);


                List<int> subtokens = new List<int>();
                for (int i = 0; i < splits.Length; i++)
                {
                    string word = splits[i];
                    if (Tokens.ContainsKey(word))
                    {
                        //data[i + 1] = Tokens[word];
                        subtokens.Add(Tokens[word]);
                    }
                    else
                    {
                        bool first = true;
                        while (word != "")
                        {
                            for (int a = word.Length - 1; a > 0; a--)
                            {
                                string sub = word.Substring(0, a);
                                string subword = first == true ? sub : $"##{sub}";
                                if (Tokens.ContainsKey(subword))
                                {
                                    //data[i + 1] = Tokens[subword];
                                    subtokens.Add(Tokens[subword]);
                                    word = Util.ReplaceFirstMatch(word, sub, "");
                                    break;
                                }
                            }
                            first = false;

                            string checksub = $"##{word}";
                            if (Tokens.ContainsKey(checksub))
                            {
                                //data[i + 1] = Tokens[word];
                                subtokens.Add(Tokens[checksub]);
                                break;
                            }

                        }
                    }
                }
                long[] data = new long[subtokens.Count + 2];
                data[0] = 101;
                data[data.Length - 1] = 102;
                for (int i = 0; i < subtokens.Count; i++)
                {
                    data[i + 1] = subtokens[i];
                }
                tokenResult.input_ids[s] = data;
                tokenResult.attention_mask[s] = ones_like(data.Length);
                tokenResult.token_type_ids[s] = new long[data.Length];
            }

            // check padding
            var dataForChecking = tokenResult.input_ids.Select(x => x.Length).ToArray();

            bool same = SameCheck(dataForChecking);
            if (same == false)
            {
                int max = dataForChecking.Max();

                for (int i = 0; i < tokenResult.input_ids.Length; i++)
                {
                    if (tokenResult.input_ids[i].Length == max)
                    {
                        continue;
                    }

                    long[] input_ids_buffered = new long[max];
                    long[] attention_mask_buffered = new long[max];
                    long[] token_type_ids_buffered = new long[max];
                    Array.Copy(tokenResult.input_ids[i], input_ids_buffered, tokenResult.input_ids[i].Length);
                    Array.Copy(tokenResult.attention_mask[i], attention_mask_buffered, tokenResult.attention_mask[i].Length);
                    tokenResult.input_ids[i] = input_ids_buffered;
                    tokenResult.attention_mask[i] = attention_mask_buffered;
                    tokenResult.token_type_ids[i] = token_type_ids_buffered;
                }
            }



            return tokenResult;
        }

        private bool SameCheck(int[] data)
        {
            if (data == null || data.Length < 2) { return true; }
            long first = data[0];
            for (int i = 0; i < data.Length; i++)
            {
                if (data[i] != first)
                {
                    return false;
                }
            }

            return true;
        }
        private long[] ones_like(int len)
        {
            long[] data = new long[len];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = 1;
            }
            return data;
        }
    }

}
