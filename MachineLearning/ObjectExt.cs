using System.IO;
using System.Text;

namespace MachineLearning;

public static class ObjectExt
{
    extension(object o)
    {
        public string Serialize() => Utils.GetXml(o, true);
        public void Serialize(string filePath) => File.WriteAllText(filePath, o.Serialize(), Encoding.Unicode);
    }

    extension<T>(T)
    {
        public static T Deserialize(string data) => Utils.LoadFromXml<T>(data);
        public static T DeserializeFile(string filePath) => Deserialize<T>(File.ReadAllText(filePath));
    }
}
