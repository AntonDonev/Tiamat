using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Utility.Services;

namespace Tiamat.Utility
{
    public class CheckPythonConnectionAttribute : ActionFilterAttribute
    {
        private readonly PythonSocketService _pythonSocketService;

        public CheckPythonConnectionAttribute(PythonSocketService pythonSocketService)
        {
            _pythonSocketService = pythonSocketService;
        }

        public override void OnActionExecuting(ActionExecutingContext context)
        {
            if (!_pythonSocketService.IsConnected)
            {
                if (context.Controller is Controller controller)
                {
                    controller.TempData["AlertMessage"] = "AI Model is Offline.";
                    controller.TempData["AlertTitle"] = "Changes cannot be pushed";
                    controller.TempData["AlertType"] = "error";
                }

                context.Result = new RedirectToActionResult("Index", "Home", null);
            }
            base.OnActionExecuting(context);
        }
    }
}
