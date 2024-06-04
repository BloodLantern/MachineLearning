using System;
using System.IO;
using System.Text;
using System.Xml;
using System.Xml.Serialization;
using MachineLearning.Models.NeuralNetwork;

namespace MachineLearning;

public static class Utils
{
    public static T LoadFromXml<T>(string xmlString, XmlSerializer serial = null)
    {
        serial ??= new(typeof(T));
        T returnValue = default;
        using StringReader reader = new(xmlString);
        object result = serial.Deserialize(reader);
        if (result is T t)
            returnValue = t;
        return returnValue;
    }

    public static string GetXml(object obj, bool omitStandardNamespaces) => GetXml(obj, null, omitStandardNamespaces);

    public static string GetXml(object obj, XmlSerializer serializer = null, bool omitStandardNamespaces = false)
    {
        XmlSerializerNamespaces ns = null;
        if (omitStandardNamespaces)
        {
            ns = new();
            ns.Add("", ""); // Disable the xmlns:xsi and xmlns:xsd lines.
        }
        using StringWriter textWriter = new();
        XmlWriterSettings settings = new() { Indent = true, IndentChars = "    ", Encoding = Encoding.Default }; // For cosmetic purposes.
        using (XmlWriter xmlWriter = XmlWriter.Create(textWriter, settings))
            (serializer ?? new XmlSerializer(obj.GetType())).Serialize(xmlWriter, obj, ns);
        return textWriter.ToString();
    }

    /// <summary>
    /// Mutates a value using the given <see cref="Random"/> instance, returning whether the value changed.
    /// </summary>
    /// <param name="random">The Random instance to use.</param>
    /// <param name="value">The value to mutate.</param>
    /// <returns>Whether the value was mutated.</returns>
    public static bool MutateValue(Random random, ref double value)
    {
        double oldValue = value;
        
        switch (random.NextDouble() * 1000.0)
        {
            case <= 2.0:
                value *= -1.0;
                break;
            case <= 4.0:
                value = random.NextDouble() - 0.5;
                break;
            case <= 6.0:
                value *= random.NextDouble() + 1.0;
                break;
            case <= 8.0:
                value *= random.NextDouble();
                break;
        }

        return Math.Abs(oldValue - value) != 0.0;
    }

    public static double Sigmoid(double value) => 1.0 / (1.0 + Math.Exp(-value));
}
