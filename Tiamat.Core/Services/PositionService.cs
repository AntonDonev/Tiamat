using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using Microsoft.Identity.Client;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tiamat.Core.Services.Interfaces;
using Tiamat.DataAccess;
using Tiamat.Models;

namespace Tiamat.Core.Services
{
    public class PositionService : IPositionService
    {
        private readonly TiamatDbContext _context;
        private readonly ILogger<PositionService> _logger;

        public PositionService(TiamatDbContext context, ILogger<PositionService> logger)
        {
            _context = context;
            _logger = logger;   
        }
        public void CreatePosition(string Symbol, string Type, Account account, decimal Size, decimal Risk, DateTime OpenedAt, string Id)
        {
            Position position = new Position();
            position.Id = Id;
            position.Symbol = Symbol;
            position.Type = Type;
            position.AccountId = account.Id;
            position.Account = account;
            position.Size = Size;
            position.Risk = Risk;
            position.Result = null;
            position.OpenedAt = OpenedAt;

            _context.Positions.Add(position);
            _context.SaveChanges();
        }

        public void ClosePosition(string Id, decimal profit, decimal currentCapital, DateTime ClosedAt, string FromIp)
        {
            var position = _context.Positions
                .Include(p => p.Account)
                .FirstOrDefault(x => x.Id == Id && x.Account.Affiliated_IP == FromIp);

            if (position == null)
            {
                _logger.LogError("No matching position found for ID {Id} with IP {FromIp}.", Id, FromIp);
                return;
            }

            position.Result = profit;

            var account = _context.Accounts
                .FirstOrDefault(x => x.Id == position.AccountId && x.Affiliated_IP == FromIp);

            if (account == null)
            {
                _logger.LogError("No matching account found for Position ID {PositionId} with IP {FromIp}.", position.AccountId, FromIp);
            }
            else
            {
                account.CurrentCapital = currentCapital;
            }

            position.ClosedAt = ClosedAt;
            _context.SaveChanges();
        }
    }
}
