using System.Diagnostics;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Tiamat.Models;
using Tiamat.WebApp.Models;

namespace Tiamat.WebApp.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        private readonly SignInManager<User> _signInManager;
        private readonly UserManager<User> _userManager;

        public HomeController(ILogger<HomeController> logger, SignInManager<User> signInManager, UserManager<User> userManager)
        {
            _logger = logger;
            _signInManager = signInManager;
            _userManager = userManager;
        }

        public IActionResult Index()
        {
            return RedirectToAction("Login", "Home");
        }

        public IActionResult Privacy()
        {
            return View();
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }

        [HttpGet]
        public IActionResult Login()
        {
            if (_signInManager.IsSignedIn(User))
            {
                TempData["AlertMessage"] = "Вие вече сте влезнали в системата. Ако искате да влезнете с друг акаунт, моля първо излезте от текущия акаунт.";
                TempData["AlertTitle"] = "Автоматично пренасочване";
                TempData["AlertType"] = "info";
                return RedirectToAction("Dashboard", "User");
            }

            return View();
        }

        [HttpPost]
        [ValidateAntiForgeryToken]
        public async Task<IActionResult> Login(string username, string password, bool rememberMe)
        {
            var result = await _signInManager.PasswordSignInAsync(username, password, rememberMe, false);
            if (result.Succeeded)
            {
                TempData["AlertMessage"] = "Потребителят влезе успешно!";
                TempData["AlertTitle"] = "Успех";
                TempData["AlertType"] = "success";
                return RedirectToAction("Dashboard", "User");
            }
            TempData["AlertMessage"] = "Неуспешен опит за влизане. Моля, проверете вашите данни.";
            TempData["AlertTitle"] = "Грешка при влизане";
            TempData["AlertType"] = "error";
            return View();
        }
    }
}
