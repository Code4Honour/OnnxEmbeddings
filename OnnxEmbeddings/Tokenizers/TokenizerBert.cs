using ICSharpCode.SharpZipLib.Checksum;
using Newtonsoft.Json.Linq;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

namespace OnnxEmbeddings.Tokenizers
{
    public class TokenizerBert
    {
        public TokenizerBert(string path, bool useJSON = false)
        {
            LoadTokenizer(path, useJSON);
        }
        private void LoadTokenizer(string path, bool useJSON = false)
        {
            if (useJSON == true)
            {
                string str = File.ReadAllText(path);
                Tokens = JsonSerializer.Deserialize<JsonObject>(str)["model"]["vocab"].Deserialize<Dictionary<string, int>>();
            }
            else
            {
                string[] vocablines = File.ReadAllLines(path);
                Tokens = new Dictionary<string, int>();
                for (int i = 0; i < vocablines.Length; i++)
                {
                    string vocab = vocablines[i];
                    if (Tokens.ContainsKey(vocab) == false)
                    {
                        Tokens.Add(vocab, i);
                    }
                }
                foreach (KeyValuePair<string, int> token in Tokens)
                {
                    if(token.Key.Length == 1)
                    {
                        if (TokensBad.ContainsKey(token.Key[0]) == false)
                        {
                            TokensBad.Add(token.Key[0], token.Value);
                        }
                    }
                    else if(token.Key.Length == 3 && token.Key.StartsWith("##"))
                    {
                        if (TokensBad.ContainsKey(token.Key[2]) == false)
                        {
                            TokensBad.Add(token.Key[2], token.Value);
                        }
                    }
                }
            }


        }
        Dictionary<string, int>? Tokens { get; set; }
        Dictionary<char, int>? TokensBad { get; set; } = new Dictionary<char, int>();

        public TokenizationResult Tokenize(string[] sentences, int start = 2, int finish = 3, string prefix = "##")
        {
            sentences = sentences.Select(x => x.Replace("\u00A0", " ").Replace("\u0020", " ").
            Replace("\u2000", " ").Replace("\u2001", " ").Replace("\u2002", " ").Replace("\u2003", " ").
            Replace("\u2004", " ").Replace("\u2005", " ").Replace("\u2006", " ").Replace("\u2007", "  ").
            Replace("\u2008", " ").Replace("\u2009", " ").Replace("\u200A", "  ").Replace("\u200B", "  ").
            Replace("\u200C", " ").Replace("\u200D", " ").Replace("\u200E", "  ").Replace("\u200F", "  ").
            Replace("\u2028", " ").Replace("\u2029", " ").Replace("\u202F", " ").Replace("\u205F", " ").
            Replace("\u2060", "").Replace("\uFEFF", " ")).ToArray();
            TokenizationResult tokenResult = new TokenizationResult();
            tokenResult.input_ids = new long[sentences.Length][];
            tokenResult.attention_mask = new long[sentences.Length][];
            tokenResult.token_type_ids = new long[sentences.Length][];
            string subPrefix = prefix;
            for (int s = 0; s < sentences.Length; s++)
            {
                string sentence = sentences[s];
                string item = Util.FullClean(sentence);
                string[] splits = item.Split(' ', StringSplitOptions.RemoveEmptyEntries);


                List<int> subtokens = new List<int>();
                for (int i = 0; i < splits.Length; i++)
                {
                    string word = splits[i];
                    if (Tokens.ContainsKey(word))
                    {
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
                                string subword = first == true ? sub : $"{subPrefix}{sub}";
                                if (Tokens.ContainsKey(subword))
                                {
                                    subtokens.Add(Tokens[subword]);
                                    word = Util.ReplaceFirstMatch(word, sub, "");
                                    break;
                                }
                            }
                            first = false;

                            var checksub = $"{subPrefix}{word}";
                            if(word == "\u00A0" || word == "\u0020")
                            {
                                break;
                            }
                            if (Tokens.ContainsKey(checksub))
                            {
                                subtokens.Add(Tokens[checksub]);
                                break;
                            }
                            else if(word.Length == 1)
                            {
                                var unused = $"{"[UNK]"}";
                                subtokens.Add(Tokens[unused]);
                                break;
                            }
                            else
                            {
                                char c = word[0];
                                if(char.IsWhiteSpace(c) == true)
                                {
                                    var unused = $"{"[UNK]"}";
                                    subtokens.Add(Tokens[unused]);
                                    break;
                                }
                            }

                        }
                    }
                }
                long[] data = new long[subtokens.Count + 2];
                data[0] = start;
                data[data.Length - 1] = finish;
                for (int i = 0; i < subtokens.Count; i++)
                {
                    data[i + 1] = subtokens[i];
                }
                tokenResult.input_ids[s] = data;
                tokenResult.attention_mask[s] = ones_like(data.Length);
                tokenResult.token_type_ids[s] = new long[data.Length];
            }

            int[] dataForChecking = tokenResult.input_ids.Select(x => x.Length).ToArray();

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
        public TokenizationResult TokenizeAlreadyProcessed(string[] sentences, int start = 101, int finish = 102)
        {
            TokenizationResult tokenResult = new TokenizationResult();

            string[][] splits = sentences.Select(x => x.Split(" ", StringSplitOptions.RemoveEmptyEntries)).ToArray();
            int max = splits.Select(x=>x.Length).Max() + 2;
            tokenResult.attention_mask = new long[sentences.Length][];
            tokenResult.token_type_ids = new long[sentences.Length][];
            tokenResult.input_ids = new long[sentences.Length][];
            int[][] indexes = new int[sentences.Length][];
       
            for (int i = 0; i < splits.Length; i++)
            {
                long[] pairs = new long[splits[i].Length];
                for (int ii = 0; ii < splits[i].Length; ii++)
                {
                    long dex = Tokens[splits[i][ii]];
                    pairs[ii] = dex;
                }
                long[] data = new long[pairs.Length + 2];
                data[0] = start;
                data[data.Length - 1] = finish;
                Array.Copy(pairs,0,data,1,pairs.Length);
                tokenResult.input_ids[i] = data;
                tokenResult.attention_mask[i] = ones_like(data.Length);
                tokenResult.token_type_ids[i] = new long[data.Length];

            }
      
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



            return tokenResult;
        }
    

        public List<TokenPair> GetRealIndexes(string doc, string prefix = "##")
        {
            doc = doc.ToLower();
            string subPrefix = prefix;
            List<TokenPair> subtokens = new List<TokenPair>();
            bool forcePrefix = false;
            for (int i = 0; i < doc.Length; i++)
            {
                forcePrefix = false;
                string word = string.Empty;
                int start = i;
                while (true)
                {
                    if (char.IsAsciiLetterOrDigit(doc[i]) == true)
                    {
                        //word = word + doc[i];
                        i++;
                        if(i >= doc.Length-1)
                        {
                            break;
                        }
                    }
                    else
                    {
                        if (start == i)
                        {
                            if (doc[i] > 32 && doc[i] < 127)
                            {
                                //word = word + doc[i];
                                i++;
                            }
                        }
                        //word = word + doc[ii];
                        break;
                    }
                }

                if(start == i)
                {
                    continue;
                }
                word = doc.Substring(start, i - start);
                i--;
            wait:
                var wordfirst = forcePrefix == true ? string.Concat(subPrefix, word) : word;
                if (Tokens.ContainsKey(wordfirst))
                {
                    subtokens.Add(new TokenPair()
                    {
                        TokenCode = Tokens[wordfirst],
                        Token = wordfirst,
                        BaseWord = wordfirst,
                        HasPrefix = forcePrefix,
                        VanillaWordIndex = i
                    });
                }
                else
                {
                    bool first = true;
                    while (word != "")
                    {
                        string sub = string.Empty;
                        bool firstSet = false;
                        for (int a = word.Length - 1; a > 0; a--)
                        {
                            sub = word.Length > 1 ? word.Substring(0, a) : word;
                            string subword = first == true ? sub : string.Concat(subPrefix, sub);
                            subword = (forcePrefix == true && first == true) ? string.Concat(subPrefix, subword) : subword;
                            if (Tokens.ContainsKey(subword))
                            {
                                subtokens.Add(new TokenPair()
                                {
                                    TokenCode = Tokens[subword],
                                    Token = subword,
                                    BaseWord = word,
                                    VanillaWordIndex = i,
                                    SubWord = subword
                                });
                                word = Util.ReplaceFirstMatch(word, sub, "");
                                firstSet = true;
                                break;
                            }
                        }
                        first = false;

                        string checksub = firstSet == true ? string.Concat(subPrefix, word) : word;
                        checksub = (forcePrefix == true && firstSet == false) ? $"{subPrefix}{checksub}" : checksub;
                        if (Tokens.ContainsKey(checksub))
                        {
                            subtokens.Add(new TokenPair() { TokenCode = Tokens[checksub], Token = checksub, BaseWord = word, VanillaWordIndex = i, SubWord = checksub });
                            break;
                        }
                        else if (firstSet == false && word.Length == 1)
                        {
                            subtokens.Add(new TokenPair()
                            {
                                TokenCode = Tokens["[UNK]"],
                                Token = "[UNK]",
                                BaseWord = word,
                                SubWord = checksub,
                                VanillaWordIndex = i
                            });
                            break;
                        }
                        else
                        {
                            if (word.Length == 1)
                            {
                                var lastcheck = string.Concat(subPrefix, word);
                                if (Tokens.ContainsKey(lastcheck))
                                {
                                    subtokens.Add(new TokenPair()
                                    {
                                        TokenCode = Tokens[lastcheck],
                                        Token = lastcheck,
                                        BaseWord = word,
                                        SubWord = lastcheck,
                                        VanillaWordIndex = i
                                    });
                                    break;
                                }
                                else
                                {
                                    subtokens.Add(new TokenPair()
                                    {
                                        TokenCode = Tokens["[UNK]"],
                                        Token = "[UNK]",
                                        BaseWord = word,
                                        SubWord = lastcheck,
                                        VanillaWordIndex = i
                                    });
                                    break;
                                }
                            }
                            else if(forcePrefix == true && firstSet == false)
                            {
                                var temp = word.Substring(0, 1);
                                var lastcheck = string.Concat(subPrefix, temp);
                                word = word.Substring(1);
                                if (Tokens.ContainsKey(lastcheck))
                                {
                                    subtokens.Add(new TokenPair()
                                    {
                                        TokenCode = Tokens[lastcheck],
                                        Token = lastcheck,
                                        BaseWord = word,
                                        SubWord = lastcheck,
                                        VanillaWordIndex = i
                                    });
                                }
                                else
                                {
                                    subtokens.Add(new TokenPair()
                                    {
                                        TokenCode = Tokens["[UNK]"],
                                        Token = "[UNK]",
                                        BaseWord = word,
                                        SubWord = lastcheck,
                                        VanillaWordIndex = i
                                    });
                                }
                            }
                            forcePrefix = true;
                            goto wait;
                        }

                    }
                }
            }

            return subtokens;
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
    [DebuggerDisplay("{Token} {BaseWord} {TokenCode} {VanillaWordIndex}")]
    public struct TokenPair
    {
        public string Token;
        public string BaseWord;
        public string SubWord;
        public bool HasPrefix;
        public int TokenCode;
        public int VanillaWordIndex;
    }

    public class FastSentence
    {
        public List<TokenPair> Tokens;
        public string Sentence;
        public int Index;
    }

}
