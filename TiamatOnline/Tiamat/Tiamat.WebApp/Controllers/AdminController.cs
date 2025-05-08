using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using System.Security.Claims;
using Tiamat.Core.Helpers;
using Tiamat.Core;
using Tiamat.Core.Services.Interfaces;
using Tiamat.Models;
using Tiamat.WebApp.Models;
using Tiamat.WebApp.Models.Admin;
using Tiamat.WebApp.Models.Account1;

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
        private readonly ILogger<AdminController> _logger;

        public AdminController(
            SignInManager<User> signInManager,
            UserManager<User> userManager,
            IAccountService accountService,
            IAccountSettingService accountSettingService,
            INotificationService notificationService,
            RoleManager<IdentityRole<Guid>> roleManager,
            IPythonApiService pythonSocketService,
            ILogger<AdminController> logger)
        {
            _signInManager = signInManager;
            _userManager = userManager;
            _accountService = accountService;
            _accountSettingService = accountSettingService;
            _notificationService = notificationService;
            _roleManager = roleManager;
            _pythonSocketService = pythonSocketService;
            _logger = logger;
        }

        [HttpGet]
        public async Task<IActionResult> Notification()
        {
            ModelState.Clear();
            return View(new NotificationViewModel());
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Notification(NotificationViewModel model)
        {
            if (!ModelState.IsValid)
                return View(model);

            var result = await _notificationService.SendNotificationToTargetsAsync(model.Title, model.Description, model.Targets);

            if (result.IsSuccess)
            {
                TempData["Success"] = result.Message;
            }
            else
            {
                TempData["Error"] = result.Message;
            }

            return RedirectToAction(nameof(Notification));
        }




        [HttpGet]
        public async Task<IActionResult> UserCenter()
        {
            return View(new UserCenterViewModel{
                RegisterModel = new RegisterUserViewModel()
            });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> UserCenter(UserCenterViewModel viewModel)
        {
            if (ModelState.IsValid)
            {
                var model = viewModel.RegisterModel;
                string randomPassword = PasswordGenerator.Generate(_userManager.Options.Password);

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

            return View(viewModel);
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
            var pendingAccounts = allAccounts.Where(a => a.Status == AccountStatus.Pending).ToList();
            return View(pendingAccounts);
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ApproveAccount(ApproveAccountViewModel model)
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
            
            var result = await _accountService.ApproveAccountAndNotifyAsync(
                model.Id, 
                model.Title, 
                model.VPSName, 
                model.AffiliatedHWID, 
                model.Message, 
                User.FindFirstValue(ClaimTypes.Email));
            
            if (!result.IsSuccess)
            {
                TempData["AlertMessage"] = "Неуспешно приемане на акаунта: " + result.ErrorMessage;
                TempData["AlertTitle"] = "Грешка при приемане";
                TempData["AlertType"] = "error";
            }
            else if (!string.IsNullOrEmpty(result.ErrorMessage))
            {
                TempData["AlertMessage"] = result.ErrorMessage;
                TempData["AlertTitle"] = "Предупреждение";
                TempData["AlertType"] = "warning";
            }
            else
            {
                TempData["AlertMessage"] = "Успех";
                TempData["AlertTitle"] = "Успех";
                TempData["AlertType"] = "success";
            }

            return RedirectToAction(nameof(AccountReview));
        }
        
        public async Task<IActionResult> DenyAccountWithNotification(DenyAccountViewModel model)
        {

            if (!ModelState.IsValid)
            {
                var errors = ModelState.Values
                    .SelectMany(v => v.Errors)
                    .Select(e => e.ErrorMessage)
                    .ToList();

                var combinedErrors = string.Join("; ", errors);

                TempData["AlertMessage"] = "Неуспешно отказване на акаунта: " + combinedErrors;
                TempData["AlertTitle"] = "Грешка при отказване";
                TempData["AlertType"] = "error";

                return RedirectToAction(nameof(AccountReview));
            }

            var result = await _accountService.DenyAccountAndNotifyAsync(
                model.Id,
                model.Title,
                model.Message,
                model.UseDefaultDenyMessage);
                
            if (!result.IsSuccess)
            {
                TempData["AlertMessage"] = "Неуспешно отказване на акаунта: " + result.ErrorMessage;
                TempData["AlertTitle"] = "Грешка при отказване";
                TempData["AlertType"] = "error";
            }
            else
            {
                TempData["AlertMessage"] = "Успех";
                TempData["AlertTitle"] = "Успех";
                TempData["AlertType"] = "success";
            }
            return RedirectToAction(nameof(AccountReview));
        }
        
        [HttpGet]
        public async Task<IActionResult> UserAccounts(string searchTerm = null, Guid? userId = null)
        {
            var model = new AccountsViewModel
            {
                SearchTerm = searchTerm
            };
            
            if (!string.IsNullOrWhiteSpace(searchTerm))
            {
                model.SearchResults = await _accountService.SearchUsersAsync(searchTerm);
            }
            
            if (userId.HasValue)
            {
                model.SelectedUserId = userId;
                var user = await _userManager.FindByIdAsync(userId.Value.ToString());
                if (user != null)
                {
                    model.SelectedUserName = user.UserName;
                    model.UserAccounts = await _accountService.GetAccountsByUserIdAsync(userId.Value);
                }
            }
            
            return View(model);
        }
        
        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> ChangeHwid(ChangeHwidViewModel model)
        {
            if (!ModelState.IsValid)
            {
                TempData["AlertMessage"] = "Неуспешна промяна на HWID: Невалидни данни.";
                TempData["AlertTitle"] = "Грешка";
                TempData["AlertType"] = "error";
                
                var account1 = await _accountService.GetAccountByIdAsync(model.AccountId);
                return RedirectToAction(nameof(UserAccounts), new { userId = account1?.UserId });
            }
            
            var result = await _accountService.ChangeAccountHwidAsync(model.AccountId, model.NewHwid);
            
            if (!result.IsSuccess)
            {
                TempData["AlertMessage"] = "Неуспешна промяна на HWID: " + result.ErrorMessage;
                TempData["AlertTitle"] = "Грешка";
                TempData["AlertType"] = "error";
            }
            else
            {
                TempData["AlertMessage"] = "HWID на акаунта е успешно променен.";
                TempData["AlertTitle"] = "Успех";
                TempData["AlertType"] = "success";
            }
            
            var account = await _accountService.GetAccountByIdAsync(model.AccountId);
            return RedirectToAction(nameof(UserAccounts), new { userId = account?.UserId });
        }
    }
}