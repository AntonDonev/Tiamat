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

        public PositionService(TiamatDbContext context)
        {
            _context = context;
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

        public void ClosePosition(string Id, decimal profit, decimal currentCapital, DateTime ClosedAt)
        {
            Position position = _context.Positions.FirstOrDefault(x => x.Id == Id);
            position.Result = profit;
            _context.Accounts.FirstOrDefault(x=>x.Id==position.AccountId).CurrentCapital = currentCapital;
            position.ClosedAt = ClosedAt;
            _context.SaveChanges();
        }
    }
}
