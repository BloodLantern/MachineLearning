using Microsoft.Xna.Framework;

namespace MonoGame.Utils.Extensions;

public static class RandomExt
{
    /// <param name="random">The Random instance</param>
    extension(Random random)
    {
        /// <summary>
        /// Returns a Vector2 that has both its component grater or equal to 0.0 and less than 1.0
        /// </summary>
        /// <returns>A random Vector2</returns>
        public Vector2 NextVector2() => new(random.NextSingle(), random.NextSingle());

        public T Choose<T>(params T[] choices) => random.Choose((IList<T>) choices);

        public T Choose<T>(IList<T> choices) => choices[random.Next(choices.Count)];

        public bool NextBoolean() => Convert.ToBoolean(random.Next(0, 1));
    }
}
