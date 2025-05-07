using System.Diagnostics;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace OnnxEmbeddings.Models
{

    [DebuggerDisplay("{DateString}\t{Title}")]
    public struct MedicalJournal
    {

        public string Title;
        public string Abstract;
        public string PMID;
        //public string AuthorListJSON { get; set; }
        public string DateString;
        //public DateTime Date { get; set; }

        public List<Author> AuthorList
        {
            get
            {
                return authorList;
            }
            set
            {
                authorList = value;
                if (value != null && value.Count > 0)
                {
                    //AuthorListJSON = JsonConvert.SerializeObject(value);
                    //AuthorListJSON = string.Join(",", value.Select(x => $"[{x.ForeName},{x.LastName}]"));
                }
            }
        }
        private List<Author> authorList;

        public byte[] ToByteArray()
        {
            // Combine all property byte arrays into one
            using (var memoryStream = new MemoryStream())
            {
                AppendProperty(memoryStream, Title);
                AppendProperty(memoryStream, Abstract);
                AppendProperty(memoryStream, PMID);
                //AppendProperty(memoryStream, AuthorListJSON);
                AppendProperty(memoryStream, DateString);

                byte[] bytes = memoryStream.ToArray();
                byte[] final = new byte[2 + bytes.Length];
                Buffer.BlockCopy(BitConverter.GetBytes((short)bytes.Length), 0, final, 0, 2);
                Buffer.BlockCopy(bytes, 0, final, 2, bytes.Length);

                return final;
            }
        }

        public static List<MedicalJournal> FromByteArray(string path,ref int offset, long bufferAmount)
        {
            FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            bufferAmount = Math.Min(stream.Length, bufferAmount);


            byte[] data = new byte[bufferAmount];

            stream.Read(data, offset, (int)bufferAmount);
            List<MedicalJournal> journals = new List<MedicalJournal>();

            while (offset < data.Length)
            {
                var journal = new MedicalJournal();
                short len = BitConverter.ToInt16(data, offset);
                if(offset + len > data.Length)
                {
                    break;
                }
                offset += 2;
                GetProperty(data, ref offset, ref journal.Title);
                GetProperty(data, ref offset, ref journal.Abstract);
                GetProperty(data, ref offset, ref journal.PMID);
                GetProperty(data, ref offset, ref journal.DateString);
                journals.Add(journal);
            }
            var tse = journals.Where(x => x.Abstract != null && x.Abstract != "").ToList();
            return journals;
          
        }

        public static List<MedicalJournal> FromByteArray(string path)
        {
            FileStream stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);

            int bufferChunk = 100_000_000;
            byte[] data = new byte[bufferChunk];
            List<MedicalJournal> journals = new List<MedicalJournal>();
            int i = 0;
            long lastGoodOffset = 0;

            while (stream.Position < stream.Length)
            {
               
                int offset = 0;
             
                DateTime d1 = DateTime.UtcNow;
                int amountRead = stream.Read(data, 0, bufferChunk);
                DateTime d2 = DateTime.UtcNow;
                var dif = (d2 - d1).TotalMilliseconds;
                Console.WriteLine(dif.ToString("N0"));
                while (true)
                {
                    if(offset >= data.Length)
                    {
                        stream.Position = lastGoodOffset;
                        break; 
                    }
                    short len = BitConverter.ToInt16(data, offset);
                    if (offset + len > amountRead)
                    {
                        stream.Position = lastGoodOffset;
                        break;
                    }
                    lastGoodOffset = stream.Position - bufferChunk + offset;

                    offset += 2;
                    MedicalJournal journal = new MedicalJournal();
                    GetProperty(data, ref offset, ref journal.Title);
                    GetProperty(data, ref offset, ref journal.Abstract);
                    GetProperty(data, ref offset, ref journal.PMID);
                    GetProperty(data, ref offset, ref journal.DateString);
                    int id = 0;
                    bool res = int.TryParse(journal.PMID, out id);
                    if(res == false)
                    {
                        //stream.Position = lastGoodOffset;
                        //break;
                    }
                    journal.Abstract = null;
                    journals.Add(journal);
                    i++;
                    if(i == 3936560 - 1)
                    {

                    }
            
                }
                if(amountRead != bufferChunk)
                {
                    break;
                }
            }
            var tse = journals.Where(x => x.Abstract != null && x.Abstract != "").ToList();
            return journals;

        }
        public static List<MedicalJournal> FromByteArrayFolder(string directory)
        {
            List<MedicalJournal> journals = new List<MedicalJournal>();
            object locker = new object();
            string[] files = Directory.GetFiles(directory);
            int maxTitleLen = 0;
            int maxAbstractLen = 0;
            files = files.Skip((int)(files.Length * 0.75)).ToArray();

            Parallel.ForEach(files, new ParallelOptions() { MaxDegreeOfParallelism = 8 }, (path) => 
            {
                List<MedicalJournal> journalsTemp = new List<MedicalJournal>();

                byte[] data = File.ReadAllBytes(path);
                int offset = 0;
                while (true)
                {
                    if (offset >= data.Length)
                    {
                        break;
                    }
                    short len = BitConverter.ToInt16(data, offset);
                    if (offset + len >= data.Length)
                    {
                        break;
                    }

                    offset += 2;
                    MedicalJournal journal = new MedicalJournal();
                    GetProperty(data, ref offset, ref journal.Title);
                    GetProperty(data, ref offset, ref journal.Abstract);
                    GetProperty(data, ref offset, ref journal.PMID);
                    GetProperty(data, ref offset, ref journal.DateString);
                    int id = 0;
                    bool res = int.TryParse(journal.PMID, out id);
                    if (res == false && journal.PMID != "-")
                    {
                        //stream.Position = lastGoodOffset;
                        break;
                    }
                    else if(journal.DateString.Length < 20)
                    {
                        if(journal.Title.Length > maxTitleLen)
                        {
                            maxTitleLen = journal.Title.Length;
                            //Console.WriteLine($"Title Length Max {maxTitleLen} \t Abstract Length Max {maxAbstractLen}");
                        }
                        if (journal.Abstract.Length > maxAbstractLen)
                        {
                            maxAbstractLen = journal.Abstract.Length;
                            //Console.WriteLine($"Title Length Max {maxTitleLen} \t Abstract Length Max {maxAbstractLen}");
                        }
                    }
                    journal.Abstract = null;
                    journalsTemp.Add(journal);
                }
                lock(locker)
                {
                    journals.AddRange(journalsTemp);
                }
            }
            );
            return journals;

        }
        private static void GetProperty(byte[] data, ref int offset, ref string propertyValue)
        {
    
            short len = BitConverter.ToInt16(data, offset);
            offset += 2;
            if(len < 0)
            {

            }
            else if (offset + len <= data.Length)
            {
                propertyValue = Encoding.UTF8.GetString(data, offset, len);
                offset += len;
            }
            else
            {

            }
        }
        private void AppendProperty(MemoryStream memoryStream, string propertyValue)
        {
            if (propertyValue == null || propertyValue.Length < 2) 
            { 
                propertyValue = "-"; 
            }
            byte[] propertyBytes = Encoding.UTF8.GetBytes(propertyValue);
            if(propertyValue == "-")
            {

            }
            if(propertyBytes.Length > short.MaxValue)
            {
                Console.WriteLine("Bad");
            }
            byte[] lengthBytes = BitConverter.GetBytes((short)propertyBytes.Length);


            // Append length and property bytes to the stream
            memoryStream.Write(lengthBytes, 0, lengthBytes.Length);
            memoryStream.Write(propertyBytes, 0, propertyBytes.Length);
        }
    }

    [XmlRoot("PubmedArticleSet")]
    public class PubmedArticleSet
    {
        [XmlElement("PubmedArticle")]
        public List<PubmedArticle> PubmedArticles { get; set; }
    }

    [XmlRoot("PubmedArticle")]
    public class PubmedArticle
    {
        [XmlElement("MedlineCitation")]
        public MedlineCitation MedlineCitation { get; set; }

        [XmlElement("PubmedData")]
        public PubmedData PubmedData { get; set; }
    }

    public class MedlineCitation
    {
        [XmlAttribute("Status")]
        public string Status { get; set; }

        [XmlAttribute("Owner")]
        public string Owner { get; set; }

        [XmlElement("PMID")]
        public Pmid PMID { get; set; }

        [XmlElement("DateCompleted")]
        public Date DateCompleted { get; set; }

        [XmlElement("DateRevised")]
        public Date DateRevised { get; set; }

        [XmlElement("Article")]
        public Article Article { get; set; }

        [XmlElement("MedlineJournalInfo")]
        public MedlineJournalInfo MedlineJournalInfo { get; set; }

        [XmlElement("ChemicalList")]
        public ChemicalList ChemicalList { get; set; }

        [XmlElement("MeshHeadingList")]
        public MeshHeadingList MeshHeadingList { get; set; }

        [XmlElement("KeywordList")]
        public KeywordList KeywordList { get; set; }

        [XmlAttribute("CoiStatement")]
        public string CoiStatement { get; set; }
    }

    public class Pmid
    {
        [XmlAttribute("Version")]
        public int Version { get; set; }

        [XmlText]
        public int Value { get; set; }
    }

    public class Date
    {
        [XmlElement("Year")]
        public string Year { get; set; }

        [XmlElement("Month")]
        public string Month { get; set; }

        [XmlElement("Day")]
        public string Day { get; set; }
    }

    public class Article
    {
        [XmlAttribute("PubModel")]
        public string PubModel { get; set; }

        [XmlElement("Journal")]
        public Journal Journal { get; set; }

        [XmlElement("ArticleTitle")]
        public string ArticleTitle { get; set; }

        [XmlElement("Pagination")]
        public Pagination Pagination { get; set; }

        [XmlElement("AuthorList")]
        public AuthorList AuthorList { get; set; }

        [XmlElement("Language")]
        public string Language { get; set; }

        [XmlElement("GrantList")]
        public GrantList GrantList { get; set; }

        [XmlElement("PublicationTypeList")]
        public PublicationTypeList PublicationTypeList { get; set; }

        //[XmlElement("Abstract")]
        //public Abstract Abstract { get; set; }
    }

    //public class Abstract
    //{
    //    [XmlElement("AbstractText")]
    //    public string AbstractText { get; set; }

    //    [XmlElement("CopyrightInformation")]
    //    public string CopyrightInformation { get; set; }
    //}

    public class Journal
    {
        [XmlElement("ISSN")]
        public Issn ISSN { get; set; }

        [XmlElement("JournalIssue")]
        public JournalIssue JournalIssue { get; set; }

        [XmlElement("Title")]
        public string Title { get; set; }

        [XmlElement("ISOAbbreviation")]
        public string ISOAbbreviation { get; set; }

   
    }

    public class Issn
    {
        [XmlAttribute("IssnType")]
        public string IssnType { get; set; }

        [XmlText]
        public string Value { get; set; }
    }

    public class JournalIssue
    {
        [XmlAttribute("CitedMedium")]
        public string CitedMedium { get; set; }

        [XmlElement("Volume")]
        public string Volume { get; set; }

        [XmlElement("Issue")]
        public string Issue { get; set; }

        [XmlElement("PubDate")]
        public Date PubDate { get; set; }
    }

    public class Pagination
    {
        [XmlElement("MedlinePgn")]
        public string MedlinePgn { get; set; }
    }

    [XmlRoot("AuthorList")]
    public class AuthorList
    {
        [XmlAttribute("CompleteYN")]
        public string CompleteYN { get; set; }

        [XmlElement("Author")]
        public List<Author> Authors { get; set; }
    }
    [XmlRoot("Author")]
    public class Author
    {
        [XmlElement("LastName")]
        [Newtonsoft.Json.JsonProperty("LN")]
        public string LastName { get; set; }

        [XmlElement("ForeName")]
        [Newtonsoft.Json.JsonProperty("FN")]
        public string ForeName { get; set; }

        [Newtonsoft.Json.JsonProperty("A")]
        public string Affiliation { get; set; }

        [XmlAttribute("ValidYN")]
        [Newtonsoft.Json.JsonIgnore]
        public string ValidYN { get; set; }

        [XmlElement("Initials")]
        [Newtonsoft.Json.JsonIgnore]
        public string Initials { get; set; }

        [XmlElement("AffiliationInfo")]
        [Newtonsoft.Json.JsonIgnore]
        public AffiliationInfo AffiliationInfo { get; set; }
     
    }

    public class AffiliationInfo
    {
        [XmlAttribute("Affiliation")]
        public string Affiliation { get; set; }
    }

    public class GrantList
    {
        [XmlAttribute("CompleteYN")]
        public string CompleteYN { get; set; }

        [XmlElement("Grant")]
        public List<Grant> Grants { get; set; }
    }

    public class Grant
    {
        [XmlElement("GrantID")]
        public string GrantID { get; set; }

        [XmlElement("Acronym")]
        public string Acronym { get; set; }

        [XmlElement("Agency")]
        public string Agency { get; set; }

        [XmlElement("Country")]
        public string Country { get; set; }
    }

    public class PublicationTypeList
    {
        [XmlElement("PublicationType")]
        public List<PublicationType> PublicationTypes { get; set; }
    }

    public class PublicationType
    {
        [XmlAttribute("UI")]
        public string UI { get; set; }

        [XmlText]
        public string Value { get; set; }
    }

    public class MedlineJournalInfo
    {
        [XmlElement("Country")]
        public string Country { get; set; }

        [XmlElement("MedlineTA")]
        public string MedlineTA { get; set; }

        [XmlElement("NlmUniqueID")]
        public string NlmUniqueID { get; set; }

        [XmlElement("ISSNLinking")]
        public string ISSNLinking { get; set; }
    }

    public class KeywordList
    {

        [XmlElement("Owner")]
        public string Owner { get; set; }

        [XmlElement("Keyword")]
        public List<Keyword> Keywords { get; set; }
    }

    public class Keyword
    {
        [XmlElement("Keyword")]
        public string KeyWord { get; set; }
    }
    public class ChemicalList
    {
        [XmlElement("Chemical")]
        public List<Chemical> Chemicals { get; set; }
    }

    public class Chemical
    {
        [XmlElement("RegistryNumber")]
        public string RegistryNumber { get; set; }

        [XmlElement("NameOfSubstance")]
        public NameOfSubstance NameOfSubstance { get; set; }
    }

    public class NameOfSubstance
    {
        [XmlAttribute("UI")]
        public string UI { get; set; }

        [XmlText]
        public string Value { get; set; }
    }

    public class MeshHeadingList
    {
        [XmlElement("MeshHeading")]
        public List<MeshHeading> MeshHeadings { get; set; }
    }

    public class MeshHeading
    {
        [XmlElement("DescriptorName")]
        public DescriptorName DescriptorName { get; set; }

        [XmlElement("QualifierName")]
        public List<QualifierName> QualifierNames { get; set; }
    }

    public class DescriptorName
    {
        [XmlAttribute("UI")]
        public string UI { get; set; }

        [XmlAttribute("MajorTopicYN")]
        public string MajorTopicYN { get; set; }

        [XmlText]
        public string Value { get; set; }
    }

    public class QualifierName
    {
        [XmlAttribute("UI")]
        public string UI { get; set; }

        [XmlAttribute("MajorTopicYN")]
        public string MajorTopicYN { get; set; }

        [XmlText]
        public string Value { get; set; }
    }

    public class PubmedData
    {
        [XmlElement("History")]
        public History History { get; set; }

        [XmlElement("PublicationStatus")]
        public string PublicationStatus { get; set; }

        [XmlElement("ArticleIdList")]
        public ArticleIdList ArticleIdList { get; set; }

        //[XmlElement("ReferenceList")]
        //public List<Reference> ReferenceList { get; set; }
    }

    public class History
    {
        [XmlElement("PubMedPubDate")]
        public List<PubMedPubDate> PubMedPubDates { get; set; }
    }

    public class PubMedPubDate
    {
        [XmlAttribute("PubStatus")]
        public string PubStatus { get; set; }

        [XmlElement("Year")]
        public int Year { get; set; }

        [XmlElement("Month")]
        public int Month { get; set; }

        [XmlElement("Day")]
        public int Day { get; set; }

        [XmlElement("Hour")]
        public int? Hour { get; set; }

        [XmlElement("Minute")]
        public int? Minute { get; set; }
    }

    public class ArticleIdList
    {
        [XmlElement("ArticleId")]
        public List<ArticleId> ArticleIds { get; set; }
    }

    public class ArticleId
    {
        [XmlAttribute("IdType")]
        public string IdType { get; set; }

        [XmlText]
        public string Value { get; set; }
    }

    public class Reference
    {
        [XmlAttribute("Citation")]
        public string Citation { get; set; }

        [XmlElement("ArticleIdList")]
        public ArticleIdList ArticleIdList { get; set; }
    }

}
