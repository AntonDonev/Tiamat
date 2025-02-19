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
        private readonly IAccountService _accountService;
        public void CreatePosition(string Symbol, string Type, Guid AccountId, decimal Size, decimal Risk, DateTime OpenedAt)
        {
            Position position = new Position();
            position.Id = Guid.NewGuid();
            position.Symbol = Symbol;
            position.Type = Type;
            position.AccountId = AccountId;
            position.Account = _accountService.GetAccountById(AccountId);
            position.Size = Size;
            position.Risk = Risk;
            position.Result = null;
            position.OpenedAt = OpenedAt;

            _context.Positions.Add(position);
            _context.SaveChanges();
        }
    }
}
