namespace OnnxEmbeddings.Tokenizers
{
    public class TokenizationResult
    {
        public long[][] input_ids { get; set; }
        public long[][] attention_mask { get; set; }
        public long[][] token_type_ids { get; set; }

        public string[] Sentences { get; set; }


        public TokenizationResult Chunk(int skip, int amount)
        {
            return new TokenizationResult { 
                input_ids = input_ids.Skip(skip).Take(amount).ToArray(), 
                attention_mask = attention_mask.Skip(skip).Take(amount).ToArray(),
                token_type_ids = token_type_ids.Skip(skip).Take(amount).ToArray(), 
            };
        }
        
        
        public long[] input_ids_flat()
        {
            int max = input_ids.Max(x => x.Length);
            long[] flat = new long[max * input_ids.Length];
            for (int i = 0; i < input_ids.Length; i++)
            {
                Array.Copy(input_ids[i], 0, flat, i * max, max);
            }
            return flat;
        }
        
        public long[] attention_mask_flat()
        {
            int max = attention_mask.Max(x => x.Length);
            long[] flat = new long[max * attention_mask.Length];
            for (int i = 0; i < attention_mask.Length; i++)
            {
                Array.Copy(attention_mask[i], 0, flat, i * max, max);
            }
            return flat;
        }
        
        public long[] token_type_ids_flat()
        {
            int max = token_type_ids.Max(x => x.Length);
            long[] flat = new long[max * token_type_ids.Length];
            for (int i = 0; i < token_type_ids.Length; i++)
            {
                Array.Copy(token_type_ids[i], 0, flat, i * max, max);
            }
            return flat;
        }
    }

}
