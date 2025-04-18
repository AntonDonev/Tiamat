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

        public async Task CreatePositionAsync(string Symbol, string Type, Account account, decimal Size, decimal Risk, DateTime OpenedAt, string Id)
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

            await _context.Positions.AddAsync(position);
            await _context.SaveChangesAsync();
        }

        public async Task ClosePositionAsync(string Id, decimal profit, decimal currentCapital, DateTime ClosedAt, string FromIp)
        {
            var position = await _context.Positions
                .Include(p => p.Account)
                .FirstOrDefaultAsync(x => x.Id == Id && x.Account.Affiliated_IP == FromIp);

            if (position == null)
            {
                _logger.LogError("No matching position found for ID {Id} with IP {FromIp}.", Id, FromIp);
                return;
            }

            position.Result = profit;

            var account = await _context.Accounts
                .FirstOrDefaultAsync(x => x.Id == position.AccountId && x.Affiliated_IP == FromIp);

            if (account == null)
            {
                _logger.LogError("No matching account found for Position ID {PositionId} with IP {FromIp}.", position.AccountId, FromIp);
            }
            else
            {
                account.CurrentCapital = currentCapital;
            }

            position.ClosedAt = ClosedAt;
            await _context.SaveChangesAsync();
        }

        public async Task<List<Position>> GetPositionsOfUserAsync(Guid userId)
        {
            var userAccounts = await _context.Accounts
                .Where(x => x.UserId == userId)
                .Select(a => a.Id)
                .ToListAsync();

            var positions = await _context.Positions
                .Where(p => userAccounts.Contains(p.AccountId))
                .ToListAsync();

            return positions;
        }

        public async Task<List<Position>> GetFilteredPositionsForAccountAsync(Guid accountId, DateTime? startDate, DateTime? endDate, string type, string resultFilter)
        {
            var query = _context.Positions.AsQueryable();
            
            query = query.Where(p => p.AccountId == accountId);
            
            if (startDate.HasValue)
            {
                query = query.Where(p => p.OpenedAt >= startDate.Value);
            }
            
            if (endDate.HasValue)
            {
                query = query.Where(p => p.OpenedAt <= endDate.Value);
            }
            
            if (!string.IsNullOrEmpty(type))
            {
                query = query.Where(p => p.Type == type);
            }
            
            if (!string.IsNullOrEmpty(resultFilter))
            {
                if (resultFilter == "profit")
                {
                    query = query.Where(p => p.Result > 0);
                }
                else if (resultFilter == "loss")
                {
                    query = query.Where(p => p.Result < 0);
                }
                else if (resultFilter == "active")
                {
                    query = query.Where(p => p.Result == null);
                }
            }
            
            return await query.OrderByDescending(p => p.OpenedAt).ToListAsync();
        }
    }
}