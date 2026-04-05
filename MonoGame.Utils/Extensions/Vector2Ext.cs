using Microsoft.Xna.Framework;
using MonoGame.Extended;

namespace MonoGame.Utils.Extensions;

public static class Vector2Ext
{
    extension(Vector2 v)
    {
        public Vector2 Normal() => new Vector2(v.Y, -v.X).NormalizedCopy();

        public static Vector2 FromAngle(float angle) => new(MathF.Cos(angle), MathF.Sin(angle));

        public float ToAngle() => MathF.Atan2(v.Y, v.X);
    }
}
