using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using System.Security.Claims;
using Tiamat.Core.Services.Interfaces;
using Tiamat.Models;

namespace Tiamat.WebApp.Controllers
{
    public class AdminController : Controller
    {
        private readonly SignInManager<User> _signInManager;
        private readonly UserManager<User> _userManager;
        private readonly IAccountService _accountService;
        private readonly IAccountSettingService _accountSettingService;
        private readonly INotificationService _notificationService;

        public AdminController(
            SignInManager<User> signInManager,
            UserManager<User> userManager,
            IAccountService accountService,
            IAccountSettingService accountSettingService,
            INotificationService notificationService)
        {
            _signInManager = signInManager;
            _userManager = userManager;
            _accountService = accountService;
            _accountSettingService = accountSettingService;
            _notificationService = notificationService;

        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult MarkNotificationAsRead(Guid notificationId)
        {
            var userIdString = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userIdString))
            {
                return Json(new { success = false });
            }

            var userId = Guid.Parse(userIdString);

            _notificationService.MarkNotificationAsRead(userId, notificationId);

            var newCount = _notificationService.GetUserUnreadNotifications(userId).Count();

            return Json(new { success = true, unreadCount = newCount });
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public IActionResult MarkAllAsRead()
        {
            var userIdString = User.FindFirstValue(ClaimTypes.NameIdentifier);
            if (string.IsNullOrEmpty(userIdString))
            {
                return Json(new { success = false });
            }

            var userId = Guid.Parse(userIdString);

            _notificationService.MarkAllNotificationsAsRead(userId);

            var newCount = _notificationService.GetUserUnreadNotifications(userId).Count();

            return Json(new { success = true, unreadCount = newCount });
        }


        [HttpGet]
        public IActionResult AccountReview()
        {
            var pendingAccounts = _accountService.GetAllAccounts()
                                   .Where(a => a.Status == AccountStatus.Pending)
                                   .ToList();

            return View(pendingAccounts);
        }

        [HttpPost]
        public IActionResult ApproveAccount(Guid id, string title, string message, bool useDefaultMessage)
        {
            _accountService.AccountReview(AccountStatus.Active, id, "asd", "asd");

            if (useDefaultMessage)
            {
                message = "Your account has been accepted...";
            }


            Notification notification = new Notification();
            notification.Id = Guid.NewGuid();
            notification.Title = title;
            notification.Description = message;
            notification.DateTime = DateTime.Now;
            List<Guid> target = new List<Guid> { _accountService.GetAccountById(id).UserId };
            _notificationService.CreateNotification(notification, target);

            return RedirectToAction(nameof(AccountReview));
        }

        [HttpPost]
        public IActionResult DenyAccountWithNotification(Guid id, string title, string message, bool useDefaultDenyMessage)
        {
            _accountService.AccountReview(AccountStatus.Active, id, "asd", "asd");


            if (useDefaultDenyMessage)
            {
                message = "After careful consideration, we regret to inform you...";
            }

            Notification notification = new Notification();
            notification.Id = Guid.NewGuid();
            notification.Title = title;
            notification.Description = message;
            notification.DateTime = DateTime.Now;
            List<Guid> target = new List<Guid> { _accountService.GetAccountById(id).UserId };
            _notificationService.CreateNotification(notification, target);

            return RedirectToAction(nameof(AccountReview));
        }
    }
}
