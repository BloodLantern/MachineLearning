using System.IO;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace MachineLearning;

public static class XmlSerializationHelper
{
    public static T LoadFromXml<T>(this string xmlString, XmlSerializer serial = null)
    {
        serial ??= new(typeof(T));
        T returnValue = default;
        using StringReader reader = new(xmlString);
        object result = serial.Deserialize(reader);
        if (result is T t)
            returnValue = t;
        return returnValue;
    }

    public static string GetXml<T>(this T obj, bool omitStandardNamespaces) => obj.GetXml(null, omitStandardNamespaces);

    public static string GetXml<T>(this T obj, XmlSerializer serializer = null, bool omitStandardNamespaces = false)
    {
        XmlSerializerNamespaces ns = null;
        if (omitStandardNamespaces)
        {
            ns = new();
            ns.Add("", ""); // Disable the xmlns:xsi and xmlns:xsd lines.
        }
        using StringWriter textWriter = new();
        XmlWriterSettings settings = new() { Indent = true, IndentChars = "    ", Encoding = Encoding.UTF8 }; // For cosmetic purposes.
        using (XmlWriter xmlWriter = XmlWriter.Create(textWriter, settings))
            (serializer ?? new XmlSerializer(obj.GetType())).Serialize(xmlWriter, obj, ns);
        return textWriter.ToString();
    }
}
