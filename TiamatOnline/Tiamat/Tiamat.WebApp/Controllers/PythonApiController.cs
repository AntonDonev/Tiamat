using Microsoft.AspNetCore.Mvc;
using Tiamat.Core.Services.Interfaces;
using Tiamat.Core;
using Tiamat.WebApp.Models.Tiamat.WebApp.ViewModels.Python;

namespace Tiamat.WebApp.Controllers
{
    [Route("api/python")]
    [ApiController]
    public class PythonApiController : ControllerBase
    {
        private readonly ILogger<PythonApiController> _logger;
        private readonly IAccountService _accountService;
        private readonly IPositionService _positionService;
        private readonly IPythonApiService _pythonApiService;

        public PythonApiController(
            ILogger<PythonApiController> logger,
            IAccountService accountService,
            IPositionService positionService,
            IPythonApiService pythonApiService)
        {
            _logger = logger;
            _accountService = accountService;
            _positionService = positionService;
            _pythonApiService = pythonApiService;
        }

        [HttpPost("open-confirm")]
        public async Task<IActionResult> OpenConfirm([FromBody] OpenConfirmRequest request)
        {
            try
            {

                var account = await _accountService.GetAccountByHwidAsync(request.FromHwid);
                if (account == null)
                {
                    _logger.LogError("Account with HWID {FromHwid} not found.", request.FromHwid);
                    return BadRequest(new { error = "Account not found" });
                }

                await _positionService.CreatePositionAsync(
                    request.Symbol,
                    request.Type == "BUY" ? "Покупка" : "Продажба",
                    account,
                    request.Size,
                    request.Risk,
                    request.OpenedAt,
                    request.Id);

                return Ok(new { status = "success" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing OPEN_CONFIRM");
                return StatusCode(500, new { error = ex.Message });
            }
        }

        [HttpPost("closed-confirm")]
        public async Task<IActionResult> ClosedConfirm([FromBody] ClosedConfirmRequest request)
        {
            try
            {
                await _positionService.ClosePositionAsync(
                    request.Id,
                    request.Profit,
                    request.CurrentCapital,
                    request.ClosedAt,
                    request.FromHwid);

                return Ok(new { status = "success" });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error processing CLOSED_CONFIRM");
                return StatusCode(500, new { error = ex.Message });
            }
        }

        [HttpGet("health")]
        public IActionResult Health()
        {
            return Ok(new { status = "healthy", timestamp = DateTime.UtcNow });
        }

        [HttpPost("start-account")]
        public async Task<IActionResult> StartAccount([FromBody] StartAccountRequest request)
        {
            try
            {
                var response = await _pythonApiService.StartAccountAsync(request.AccountId, request.Hwid);
                if (response.IsSuccess)
                {
                    return Ok(new { status = "success" });
                }
                return BadRequest(new { error = response.ErrorMessage });
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error sending START command to Python");
                return StatusCode(500, new { error = ex.Message });
            }
        }
    }
}
