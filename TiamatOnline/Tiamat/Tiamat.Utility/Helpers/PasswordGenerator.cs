using Microsoft.AspNetCore.Identity;

namespace Tiamat.Utility.Helpers
{
    public static class PasswordGenerator
    {
        public static string Generate(PasswordOptions options, int length = 12)
        {
            bool requireDigit = options.RequireDigit;
            bool requireLowercase = options.RequireLowercase;
            bool requireUppercase = options.RequireUppercase;
            bool requireNonAlphanumeric = options.RequireNonAlphanumeric;

            string digitChars = "0123456789";
            string lowerChars = "abcdefghijklmnopqrstuvwxyz";
            string upperChars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
            string nonAlpha = "!@#$%^&*()-_=+[]{}<>?";

            var charPool = new List<char>();
            if (requireLowercase) charPool.AddRange(lowerChars);
            if (requireUppercase) charPool.AddRange(upperChars);
            if (requireDigit) charPool.AddRange(digitChars);
            if (requireNonAlphanumeric) charPool.AddRange(nonAlpha);

            if (!charPool.Any())
                charPool.AddRange(lowerChars + upperChars + digitChars + nonAlpha);

            var rnd = new Random();
            var passwordChars = new List<char>();

            if (requireDigit)
                passwordChars.Add(digitChars[rnd.Next(digitChars.Length)]);
            if (requireLowercase)
                passwordChars.Add(lowerChars[rnd.Next(lowerChars.Length)]);
            if (requireUppercase)
                passwordChars.Add(upperChars[rnd.Next(upperChars.Length)]);
            if (requireNonAlphanumeric)
                passwordChars.Add(nonAlpha[rnd.Next(nonAlpha.Length)]);

            while (passwordChars.Count < length)
            {
                passwordChars.Add(charPool[rnd.Next(charPool.Count)]);
            }

            return new string(passwordChars.OrderBy(_ => rnd.Next()).ToArray());
        }
    }
}