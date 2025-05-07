using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OnnxEmbeddings.Models
{
    public enum EmbeddingModels
    {
        [DownloadFlag("https://huggingface.co/BAAI/bge-m3/resolve/main/onnx/model.onnx?download=true", "https://huggingface.co/BAAI/bge-m3/resolve/main/onnx/model.onnx_data?download=true")]
        [EnumFlag(@"embeddings\bge-m3\bge-m3.onnx", @"embeddings\bge-m3\model.onnx_data")]
        bge_m3,
        /// <summary>
        /// Output - Batch, Input Dim, 1024
        /// </summary>
        [DownloadFlag("https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\bge-large-en-v1.5\bge-large-en-v1.5.onnx")]
        bge_large_en_v1_5,
        /// <summary>
        /// Output - Batch, Input Dim, 768
        /// </summary>
        [DownloadFlag("https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\bge-base-en-v1.5\bge-base-en-v1.5.onnx",@"embeddings\jina-embeddings-v3\model.onnx_data")]
        bge_base_en_v1_5,


        /// <summary>
        /// Output - Batch, Input Dim, 1024
        /// Input Variable
        /// </summary>
        [DownloadFlag("https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx?download=true", "https://huggingface.co/jinaai/jina-embeddings-v3/resolve/main/onnx/model.onnx_data?download=true")]
        [EnumFlag(@"embeddings\jina-embeddings-v3\model.onnx")]
        jina_embeddings_v3_570M,
        /// <summary>
        /// Output - Batch, Input Dim, 768
        /// </summary>
        [DownloadFlag("https://huggingface.co/jinaai/jina-embeddings-v2-base-en/resolve/main/model.onnx?download=true")]
        [EnumFlag(@"embeddings\jina-embeddings-v2-base-en\model.onnx")]
        jina_embeddings_v2_base_en_137M,
        /// <summary>
        /// Output - Batch, Input Dim, 512
        /// </summary>
        [DownloadFlag("https://huggingface.co/jinaai/jina-embeddings-v2-small-en/resolve/main/model.onnx?download=true")]
        [EnumFlag(@"embeddings\jina-embeddings-v2-small-en\model.onnx")]
        jina_embeddings_v2_small_en_33M,
        [DownloadFlag("https://huggingface.co/jinaai/jina-embeddings-v2-base-code/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\jina-embeddings-v2-base-code\model.onnx")]
        jina_embeddings_v2_base_code_161M,


        /// <summary>
        /// Output - Batch, Input Dim, 384
        /// </summary>
        [DownloadFlag("ttps://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\all-MiniLM-L12-v2\all-MiniLM-L12-v2.onnx")]
        all_MiniLM_L12_v2,
        /// <summary>
        /// Output - Batch, Input Dim, 384
        /// </summary>
        [DownloadFlag("https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\all-MiniLM-L6-v2\all-MiniLM-L6-v2.onnx")]
        all_MiniLM_L6_v2,
        [DownloadFlag("https://huggingface.co/sentence-transformers/all-mpnet-base-v2/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\all-mpnet-base-v2\model.onnx")]
        all_mpnet_base_v2,


        /// <summary>
        /// Output - Batch, Input Dim, 1024
        /// </summary>
        [DownloadFlag("https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\mxbai-embed-large-v1\mxbai-embed-large-v1.onnx")]
        mxbai_embed_large_v1,
        /// <summary>
        /// Output - Batch, Input Dim, 384
        /// </summary>
        [DownloadFlag("https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\mxbai-embed-xsmall-v1\model.onnx")]
        mxbai_embed_xsmall_v1,


        /// <summary>
        /// Output - Batch, Input Dim, 768
        /// Input Variable
        /// </summary>
        [DownloadFlag("https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\nomic-embed-text-v1.5\model.onnx")]
        nomic_embed_text_v1_5,
        /// <summary>
        /// Output - Batch, Input Dim, 384
        /// </summary>
        [DownloadFlag("https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\snowflake-arctic-embed-xs\model.onnx")]
        snowflake_arctic_embed_xs_22M,
        /// <summary>
        /// Output - Batch, Input Dim, 384
        /// </summary>
        [DownloadFlag("https://huggingface.co/Snowflake/snowflake-arctic-embed-s/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\snowflake-arctic-embed-s\model.onnx")]
        snowflake_arctic_embed_s_33M,
        /// <summary>
        /// Output - Batch, Input Dim, 768
        /// </summary>
        [DownloadFlag("https://huggingface.co/Snowflake/snowflake-arctic-embed-m/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\snowflake-arctic-embed-m\model.onnx")]
        snowflake_arctic_embed_m_110M,
        /// <summary>
        /// Output - Batch, Input Dim, 1024
        /// </summary>
        [DownloadFlag("https://huggingface.co/Snowflake/snowflake-arctic-embed-l/resolve/main/onnx/model.onnx?download=true")]
        [EnumFlag(@"embeddings\snowflake-arctic-embed-l\model.onnx")]
        snowflake_arctic_embed_l_335M,
        
        
        [EnumFlag(@"embeddings\medembed-small\model.onnx")]
        medembed_small,
        [EnumFlag(@"embeddings\S-PubMedBert-MS-MARCO\S-PubMedBert-MS-MARCO.onnx")]
        S_PubMedBert_MS_MARCO,
        [EnumFlag(@"embeddings\pubmedbert-base-embeddings\pubmedbert-base-embeddings.onnx")]
        pubmedbert_base_embeddings,
        
        
        [EnumFlag(@"embeddings\legal-roberta-base\legal-roberta-base.onnx")]
        legal_roberta_base,
        [EnumFlag(@"embeddings\legal_roberta_base_uncase\legal_roberta_base_uncase.onnx")]
        legal_roberta_base_uncase,

        /// <summary>
        /// To do
        /// </summary>
        codet5p_110m_embedding,
        /// <summary>
        /// To do
        /// </summary>
        codebert_base
    }
    [AttributeUsage(AttributeTargets.Field)]
    internal class EnumFlagAttribute : Attribute
    {
        internal string Name { get; }
        internal string Aux { get; }
        internal EnumFlagAttribute(string name, string aux = null)
        {
            Name = name;
            Aux= aux;
        }
    }
    [AttributeUsage(AttributeTargets.Field)]
    internal class DownloadFlagAttribute : Attribute
    {
        internal string Name { get; }
        internal string Aux { get; }
        internal DownloadFlagAttribute(string name, string aux = null)
        {
            Name = name;
            Aux = aux;
        }
    }


    public static class EnumExtensions
    {
        public static string GetEnumFlagValue(this Enum enumValue)
        {
            try
            {
                Type type = enumValue.GetType();
                System.Reflection.MemberInfo[] memberInfo = type.GetMember(enumValue.ToString());
                EnumFlagAttribute? attribute = (EnumFlagAttribute)Attribute.GetCustomAttribute(memberInfo[0], typeof(EnumFlagAttribute));

                return attribute?.Name;
            }
            catch (Exception e)
            {
                return null;
            }
        }
        public static string GetEnumFlagValueAux(this Enum enumValue)
        {
            try
            {
                Type type = enumValue.GetType();
                System.Reflection.MemberInfo[] memberInfo = type.GetMember(enumValue.ToString());
                EnumFlagAttribute? attribute = (EnumFlagAttribute)Attribute.GetCustomAttribute(memberInfo[0], typeof(EnumFlagAttribute));

                return attribute?.Aux;
            }
            catch (Exception e)
            {
                return null;
            }
        }
    }
}
