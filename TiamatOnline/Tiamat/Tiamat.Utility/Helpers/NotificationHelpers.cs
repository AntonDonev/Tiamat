using System.Text.RegularExpressions;

namespace Tiamat.Utility.Helpers
{
    public static class NotificationHelpers
    {
        private static readonly Regex MentionRegex = new Regex(@"@([^\s,;]+)", RegexOptions.Compiled);
        public static List<string> ExtractMentions(string text)
        {
            var results = new List<string>();
            if (string.IsNullOrWhiteSpace(text))
                return results;

            var matches = MentionRegex.Matches(text);
            foreach (Match match in matches)
            {
                if (match.Success && match.Groups.Count > 1)
                {
                    results.Add(match.Groups[1].Value.ToLower().Trim());
                }
            }
            return results;
        }
    }
}