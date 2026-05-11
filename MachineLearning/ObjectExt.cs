using System.IO;
using System.Text;
using MessagePack;

namespace MachineLearning;

public static class ObjectExt
{
    extension(object o)
    {
        public string SerializeXml() => Utils.GetXml(o, true);
        public void SerializeXml(string filePath) => File.WriteAllText(filePath, o.SerializeXml(), Encoding.Unicode);
        public byte[] SerializeBinary(MessagePackSerializerOptions options = null) => MessagePackSerializer.Serialize(o, options);
        public void SerializeBinary(string filePath, MessagePackSerializerOptions options = null) => File.WriteAllBytes(filePath, o.SerializeBinary(options));
    }

    extension<T>(T)
    {
        public static T DeserializeXml(string data) => Utils.LoadFromXml<T>(data);
        public static T DeserializeXmlFile(string filePath) => DeserializeXml<T>(File.ReadAllText(filePath));
        public static T DeserializeBinary(byte[] data, MessagePackSerializerOptions options = null) => MessagePackSerializer.Deserialize<T>(data, options);
        public static T DeserializeBinaryFile(string filePath, MessagePackSerializerOptions options = null) => DeserializeBinary<T>(File.ReadAllBytes(filePath), options);
    }
}
