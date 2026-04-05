using Microsoft.Xna.Framework;

namespace MonoGame.Utils;

public static class Calc
{
    public static float RemapValue(float oldValue, Vector2 oldRange, Vector2 newRange)
        => (oldValue - oldRange.X) * (newRange.Y - newRange.X) / (oldRange.Y - oldRange.X) + newRange.X;

    public static float ComputeDifference(float value, float targetValue, float maximumValue, float factor)
        => factor - Math.Abs(targetValue - value) / maximumValue * factor;

    public static bool LineIntersects(Vector2 l0, Vector2 l1, Vector2 point, float range = 0.2f)
        => FloatEquals((point - l1).Length() + (point - l0).Length(), (l1 - l0).Length(), range);

    public static bool FloatEquals(float a, float b, float tolerance = 1e-5f)
        => MathF.Abs(a - b) < tolerance;

    /// <summary>
    /// Normalizes a radiant angle in the range [0; 2 * Pi]
    /// </summary>
    /// <param name="angle">The radiant angle to normalize.</param>
    /// <returns>The normalized radiant angle.</returns>
    public static float NormalizeRadiantAngle(float angle) => (angle + MathHelper.TwoPi) % MathHelper.TwoPi;
}
