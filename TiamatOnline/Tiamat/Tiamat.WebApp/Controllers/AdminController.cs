using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;
using System.Text.RegularExpressions;
using Tiamat.Core.Services.Interfaces;
using Tiamat.Models;
using Tiamat.Utility;
using Tiamat.Utility.Services;
using Tiamat.WebApp.Models;

namespace Tiamat.WebApp.Controllers
{
    [Authorize(Roles = "admin")]
    public class AdminController : Controller
    {
        private readonly SignInManager<User> _signInManager;
        private readonly UserManager<User> _userManager;
        private readonly IAccountService _accountService;
        private readonly RoleManager<IdentityRole<Guid>> _roleManager;
        private readonly IAccountSettingService _accountSettingService;
        private readonly INotificationService _notificationService;
        private readonly IPythonApiService _pythonSocketService;

        public AdminController(
            SignInManager<User> signInManager,
            UserManager<User> userManager,
            IAccountService accountService,
            IAccountSettingService accountSettingService,
            INotificationService notificationService,
            RoleManager<IdentityRole<Guid>> roleManager,
            IPythonApiService pythonSocketService)
        {
            _signInManager = signInManager;
            _userManager = userManager;
            _accountService = accountService;
            _accountSettingService = accountSettingService;
            _notificationService = notificationService;
            _roleManager = roleManager;
            _pythonSocketService = pythonSocketService;
        }

        [HttpGet]
        public async Task<IActionResult> Notification()
        {
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Notification(NotificationViewModel model)
        {
            if (!ModelState.IsValid)
                return RedirectToAction(nameof(Notification));

            var notification = new Notification
            {
                Title = model.Title,
                Description = model.Description
            };

            var mentions = NotificationHelpers.ExtractMentions(model.Targets);

            mentions = mentions.Distinct(StringComparer.OrdinalIgnoreCase).ToList();

            if (mentions.Contains("everyone", StringComparer.OrdinalIgnoreCase))
            {
                await _notificationService.CreateNotificationEveryoneAsync(notification);
                TempData["Success"] = "Нотификацията е изпратена до всички!";
                return RedirectToAction(nameof(Notification));
            }

            var userIds = new List<Guid>();

            foreach (var mention in mentions)
            {
                var byUserName = await _userManager.FindByNameAsync(mention);
                if (byUserName != null)
                {
                    userIds.Add(byUserName.Id);
                    continue;
                }

                var byEmail = await _userManager.FindByEmailAsync(mention);
                if (byEmail != null)
                {
                    userIds.Add(byEmail.Id);
                    continue;
                }
            }

            userIds = userIds.Distinct().ToList();

            if (userIds.Count > 0)
            {
                await _notificationService.CreateNotificationAsync(notification, userIds);
                TempData["Success"] = $"Нотификацията е изпратена до {userIds.Count} човека!";
            }
            else
            {
                TempData["Error"] = "Няма валидни хора намерени.";
            }

            return RedirectToAction(nameof(Notification));
        }


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

        [HttpGet]
        public async Task<IActionResult> UserCenter()
        {
            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> UserCenter(RegisterUserViewModel model)
        {
            if (ModelState.IsValid)
            {
                string randomPassword = GenerateRandomPassword();

                var newUser = new User
                {
                    UserName = model.UserName,
                    Email = model.Email,
                    EmailConfirmed = true
                };

                var result = await _userManager.CreateAsync(newUser, randomPassword);

                if (result.Succeeded)
                {
                    if (!await _roleManager.RoleExistsAsync("normal"))
                    {
                        await _roleManager.CreateAsync(new IdentityRole<Guid> { Name = "normal" });
                    }

                    if (!await _userManager.IsInRoleAsync(newUser, "normal"))
                    {
                        await _userManager.AddToRoleAsync(newUser, "normal");
                    }

                    TempData["UserCreated"] = true;
                    TempData["GeneratedPassword"] = randomPassword;

                    return RedirectToAction(nameof(UserCenter));
                }
                else
                {
                    foreach (var error in result.Errors)
                    {
                        ModelState.AddModelError(string.Empty, error.Description);
                    }
                }
            }

            return View(model);
        }

        private string GenerateRandomPassword()
        {
            var options = _userManager.Options.Password;

            int length = 12;

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

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> MarkNotificationAsRead(Guid notificationId)
        {
            var userIdString = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userIdString))
            {
                return Json(new { success = false });
            }

            var userId = Guid.Parse(userIdString);

            await _notificationService.MarkNotificationAsReadAsync(userId, notificationId);

            var newCount = (await _notificationService.GetUserUnreadNotificationsAsync(userId)).Count();

            return Json(new { success = true, unreadCount = newCount });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> MarkAllAsRead()
        {
            var userIdString = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userIdString))
            {
                return Json(new { success = false });
            }

            var userId = Guid.Parse(userIdString);

            await _notificationService.MarkAllNotificationsAsReadAsync(userId);

            var newCount = (await _notificationService.GetUserUnreadNotificationsAsync(userId)).Count();

            return Json(new { success = true, unreadCount = newCount });
        }

        [HttpGet]
        public async Task<IActionResult> AccountReview()
        {
            var allAccounts = await _accountService.GetAllAccountsAsync();
            var pendingAccounts = allAccounts
                                   .Where(a => a.Status == AccountStatus.Pending)
                                   .ToList();

            return View(pendingAccounts);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        [ServiceFilter(typeof(CheckPythonConnectionAttribute))]
        public async Task<IActionResult> ApproveAccount(Guid id, string title, string message, bool useDefaultMessage, string VPSName, string AffiliatedIP)
        {
            if (!ModelState.IsValid)
            {
                var errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList();

                var combinedErrors = string.Join("; ", errors);

                TempData["AlertMessage"] = "Неуспешно приемане на акаунта: " + combinedErrors;
                TempData["AlertTitle"] = "Грешка при приемане";
                TempData["AlertType"] = "error";
                ViewBag.AccountSettings = (await _accountSettingService.GetSettingsForUserAsync(Guid.Empty)).ToList();

                return RedirectToAction(nameof(AccountReview));
            }
            await _accountService.AccountReviewAsync(AccountStatus.Active, id, VPSName, User.FindFirstValue(ClaimTypes.Email), AffiliatedIP);

            Notification notification = new Notification();
            notification.Id = Guid.NewGuid();
            notification.Title = title;
            notification.Description = message;
            notification.DateTime = DateTime.Now;

            var account = await _accountService.GetAccountByIdAsync(id);
            List<Guid> target = new List<Guid> { account.UserId };

            await _notificationService.CreateNotificationAsync(notification, target);
            
            await _pythonSocketService.StartAccountAsync(account.Id.ToString(),account.Affiliated_IP);

            return RedirectToAction(nameof(AccountReview));
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> DenyAccountWithNotification(Guid id, string title, string message, bool useDefaultDenyMessage)
        {

            await _accountService.AccountReviewAsync(AccountStatus.Active, id);

            if (useDefaultDenyMessage)
            {
                message = "After careful consideration, we regret to inform you...";
            }

            Notification notification = new Notification();
            notification.Id = Guid.NewGuid();
            notification.Title = title;
            notification.Description = message;
            notification.DateTime = DateTime.Now;

            var account = await _accountService.GetAccountByIdAsync(id);
            List<Guid> target = new List<Guid> { account.UserId };

            await _notificationService.CreateNotificationAsync(notification, target);

            return RedirectToAction(nameof(AccountReview));
        }
    }
}
