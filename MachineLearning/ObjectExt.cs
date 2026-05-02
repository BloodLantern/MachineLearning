using System.IO;
using System.Text;

namespace MachineLearning;

public static class ObjectExt
{
    extension(object o)
    {
        public void Serialize(string filePath) => File.WriteAllText(filePath, Utils.GetXml(o, true), Encoding.Unicode);
    }

    extension<T>(T)
    {
        public static T Deserialize(string filePath) => Utils.LoadFromXml<T>(File.ReadAllText(filePath));
    }
}
