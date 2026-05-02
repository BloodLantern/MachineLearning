namespace MonoGame.Utils.Extensions;

public static class IListExt
{
    extension<T>(IList<T> iList)
    {
        public T Random() => iList.Random(System.Random.Shared);
        public T Random(Random random) => random.Choose(iList);
    }
}
