using OnnxEmbeddings;
using OnnxEmbeddings.Models;
using OnnxEmbeddings.Tokenizers;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;

public class TextProcessor
{
    public struct ReplacementInfo
    {
        public int Index;
        public string OriginalString;
        public char OriginalChar;
        public bool Ignore;
        public ReplacementInfo(int index, string originalString)
        {
            Index = index;
            OriginalString = originalString;
        }
        public ReplacementInfo(int index, char originalChar)
        {
            Index = index;
            OriginalChar = originalChar;
        }
        public ReplacementInfo(int index, char originalChar, bool ignore)
        {
            Index = index;
            OriginalChar = originalChar;
            Ignore = ignore;
        }
    }

    public static (string cleaned, List<ReplacementInfo> replacements) RegexClean(string sentence)
    {
        var replacements = new List<ReplacementInfo>();
        var cleanedChars = new List<char>();

        for (int i = 0; i < sentence.Length; i++)
        {
            switch (sentence[i])
            {
                case '\r':
                    replacements.Add(new ReplacementInfo(cleanedChars.Count, "\r"));
                    cleanedChars.Add(' '); // Add the character to the cleaned string
                    break;
                case '\n':
                    replacements.Add(new ReplacementInfo(cleanedChars.Count, "\n"));
                    cleanedChars.Add(' '); // Add the character to the cleaned string
                    break;
                case '\t':
                    replacements.Add(new ReplacementInfo(cleanedChars.Count, "\t"));
                    cleanedChars.Add(' '); // Add the character to the cleaned string
                    break;
                default:
                    cleanedChars.Add(sentence[i]);
                    break;
            }
        }

        return (new string(cleanedChars.ToArray()), replacements);
    }
    public static (string cleaned, ReplacementInfo[] replacements) RegexCleanFull(string sentence)
    {
        var replacements = new ReplacementInfo[sentence.Length];
        var cleanedChars = new char [sentence.Length];
        for (int i = 0; i < sentence.Length; i++)
        {
            if (sentence[i] == '\r')
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, "\r") ;
                cleanedChars[i] = ' '; // Add the character to the cleaned string
            }
            else if (sentence[i] == '\n')
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, "\n");
                cleanedChars[i] = ' '; // Add the character to the cleaned string
            }
            else if (sentence[i] == '\t')
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, "\t");
                cleanedChars[i] = ' '; // Add the character to the cleaned string
            }
            else if (char.IsLetterOrDigit(sentence[i]) == false)
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, sentence[i]);
                //replacements[i] = new ReplacementInfo(cleanedChars.Length, Substring(sentence,i));
                //    cleanedChars[i] = $" {sentence[i]} "; // Add the character to the cleaned string
            }
            else
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, sentence[i]);
                //replacements[i] = new ReplacementInfo(cleanedChars.Length, sentence.Substring(i, 1));
                //replacements[i] = new ReplacementInfo(cleanedChars.Length, Substring(sentence, i));
                cleanedChars[i] = sentence[i]; // Add the character to the cleaned string

            }
        }

        return (new string(cleanedChars), replacements);
    }
    public static (string cleaned, ReplacementInfo[] replacements) RegexCleanFull2(string sentence)
    {
        var replacements = new List<ReplacementInfo>();
        var cleanedChars = new List<char>();
        for (int i = 0; i < sentence.Length; i++)
        {
            if (sentence[i] == '\r')
            {
                replacements.Add(new ReplacementInfo(cleanedChars.Count, '\r'));
                cleanedChars.Add(' '); // Add the character to the cleaned string
            }
            else if (sentence[i] == '\n')
            {
                replacements.Add(new ReplacementInfo(cleanedChars.Count, '\n'));
                cleanedChars.Add(' '); // Add the character to the cleaned string
            }
            else if (sentence[i] == '\t')
            {
                replacements.Add(new ReplacementInfo(cleanedChars.Count, '\t'));
                cleanedChars.Add(' '); // Add the character to the cleaned string
            }
            else if (sentence[i] == ' ')
            {
                replacements.Add(new ReplacementInfo(cleanedChars.Count, ' '));
                cleanedChars.Add(' '); // Add the character to the cleaned string
            }
            else if (char.IsLetterOrDigit(sentence[i]) == false)
            {
                replacements.Add(new ReplacementInfo(cleanedChars.Count, sentence[i],true));
                replacements.Add(new ReplacementInfo(replacements.Count, sentence[i]));
                replacements.Add(new ReplacementInfo(replacements.Count, sentence[i],true));
                //replacements[i] = new ReplacementInfo(cleanedChars.Length, Substring(sentence,i));
                //    cleanedChars[i] = $" {sentence[i]} "; // Add the character to the cleaned string
                cleanedChars.Add(' '); // Add the character to the cleaned string
                cleanedChars.Add(sentence[i]); // Add the character to the cleaned string
                cleanedChars.Add(' '); // Add the character to the cleaned string
            }
            else
            {
                replacements.Add(new ReplacementInfo(cleanedChars.Count, sentence[i]));
                //replacements[i] = new ReplacementInfo(cleanedChars.Length, sentence.Substring(i, 1));
                //replacements[i] = new ReplacementInfo(cleanedChars.Length, Substring(sentence, i));
                cleanedChars.Add(sentence[i]); // Add the character to the cleaned string

            }
        }

        return (new string(cleanedChars.ToArray()), replacements.ToArray());
    }

    public static (string cleaned, ReplacementInfo[] replacements) RegexCleanFullParallel(string sentence)
    {
        var replacements = new ReplacementInfo[sentence.Length];
        var cleanedChars = new string[sentence.Length];
        Parallel.For(0, sentence.Length, new ParallelOptions() { MaxDegreeOfParallelism = 4}, (i) =>
        {
            if (sentence[i] == '\r')
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, "\r");
                cleanedChars[i] = " "; // Add the character to the cleaned string
            }
            else if (sentence[i] == '\n')
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, "\n");
                cleanedChars[i] = " "; // Add the character to the cleaned string
            }
            else if (sentence[i] == '\t')
            {
                replacements[i] = new ReplacementInfo(cleanedChars.Length, "\t");
                cleanedChars[i] = " "; // Add the character to the cleaned string
            }
            else if (char.IsLetterOrDigit(sentence[i]) == false)
            {
                //replacements[i] = new ReplacementInfo(cleanedChars.Length, sentence.Substring(i, 1));
                //cleanedChars[i] = $" {sentence[i]} "; // Add the character to the cleaned string
            }
            else
            {

                //replacements[i] = new ReplacementInfo(cleanedChars.Length, sentence.Substring(i, 1));
                //cleanedChars[i] = $"{sentence[i]}"; // Add the character to the cleaned string

            }
        });

        return (string.Join("", cleanedChars), replacements);
    }

    public static string Reconstruct(string cleaned, List<ReplacementInfo> replacements)
    {
        var reconstructedChars = new List<char>(cleaned);

        // Sort the replacements by index in descending order to avoid index shifting issues
        replacements = replacements.OrderByDescending(r => r.Index).ToList();

        foreach (var replacement in replacements)
        {
            reconstructedChars[replacement.Index] = replacement.OriginalString[0];
        }

        return new string(reconstructedChars.ToArray());
    }
    public static string ReconstructFull(string cleaned, ReplacementInfo[] replacements)
    {
        var reconstructedChars = new List<char>(cleaned);

        // Sort the replacements by index in descending order to avoid index shifting issues
        replacements = replacements.OrderByDescending(r => r.Index).ToArray();

        foreach (var replacement in replacements)
        {
            reconstructedChars[replacement.Index] = replacement.OriginalString[0];
        }

        return new string(reconstructedChars.ToArray());
    }
    public static string ReconstructFullWorking(string cleaned, ReplacementInfo[] replacements)
    {
        var reconstructedChars = new List<char>();

        // Sort the replacements by index in descending order to avoid index shifting issues

        foreach (var replacement in replacements)
        {
            if(replacement.Ignore == false)
            {
                reconstructedChars.Add(replacement.OriginalChar);

            }
        }

        return new string(reconstructedChars.ToArray());
    }
    internal static void Test()
    {
        string input = File.ReadAllText("test.txt");
        (string modelName, string folder) = Util.GetModelNameAndFolder(EmbeddingModels.mxbai_embed_large_v1);
        TokenizerBert tokenizer = new TokenizerBert(Path.Combine(folder, "vocab.txt"));
        List<TokenPair> adsf = tokenizer.GetRealIndexes(input);


        var comp = input.Replace("\r", " ").Replace("\n", " ").Replace("\t", " ");
        comp = Regex.Replace(comp, @"[^a-zA-Z0-9\s]", " $0 ");
        var avgs = new List<double>();
        while (true)
        {
            var d1 = DateTime.UtcNow;
            //var f = RegexCleanFull2(input);
            //(string cleaned, ReplacementInfo[] replacements) = RegexCleanFull2(input);
            //string[] splits = cleaned.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            //(List<FastSentence> final, List<List<TokenPair>> finalPairs) = BatchTokenizing.CompleteDocumentEmbeddingWorking(splits);
            var adsdf = tokenizer.GetRealIndexes(input);
            //while (true) 
            //{

            //}
            //bool same0 = comp == cleaned;
            //string reconstructed = ReconstructFullWorking(cleaned, replacements);
            //bool same = input == reconstructed;
            var dif = (DateTime.UtcNow - d1).TotalMilliseconds;
            avgs.Add(dif); if(avgs.Count > 40) { avgs.RemoveAt(0); }
            Console.WriteLine($"{avgs.Average().ToString("N3")}");

        }

    }
}