using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Tiamat.Utility
{
    public class CheckPythonConnectionAttribute : ActionFilterAttribute
    {
        private readonly IPythonApiService _pythonApiService;

        public CheckPythonConnectionAttribute(IPythonApiService pythonApiService)
        {
            _pythonApiService = pythonApiService;
        }

        public override void OnActionExecuting(ActionExecutingContext context)
        {
            if (!_pythonApiService.ConnectionState)
            {
                if (context.Controller is Controller controller)
                {
                    controller.TempData["AlertMessage"] = "ИИ е офлайн.";
                    controller.TempData["AlertTitle"] = "Промените не могат да бъдат запазени.";
                    controller.TempData["AlertType"] = "error";
                }

                context.Result = new RedirectToActionResult("Index", "Home", null);
            }
            base.OnActionExecuting(context);
        }
    }
}
