from cProfile import label
from faulthandler import disable
from gettext import install
from numpy.lib.arraysetops import setdiff1d
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

pair = 'btcusdt'

@dataclass
class Order:  # Our own placed order
    def __init__(self):
        pass
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class AnonTrade:  # Market trade
    def __init__(self, timestamp: float, side: str,  size: float,  price: str):
      self.timestamp = timestamp
      self.side = side
      self.size = size
      self.price = price
    timestamp: float
    side: str
    size: float
    price: str


@dataclass
class OwnTrade:  # Execution of own placed order
    def __init__(self):
        pass
    timestamp: float
    trade_id: int
    order_id: int
    side: str
    size: float
    price: float


@dataclass
class OrderbookSnapshotUpdate:  # Orderbook tick snapshot
    def __init__(self):
        pass
    timestamp: float
    asks: list[tuple[float, float]]  # tuple[price, size]
    bids: list[tuple[float, float]]


@dataclass
class MdUpdate:  # Data of a tick
    def __init__(self, md: pd.core.frame.DataFrame, timestamp: float, delta_t: float):   
      
      self.time_to_next_tick = delta_t
      
      self.trades = list(map(lambda i: AnonTrade(timestamp, md['aggro_side'].values[i],
                                      md['size'].values[i], md['price'].values[i] ) , 
                                      range(len(md['size']))))
      self.recieve_trades_latency = list(map(lambda i: (-md['exchange_ts']+md['receive_ts_x']).values[i], 
                                             range(len(md['size']))))
      
      self.orderbook = OrderbookSnapshotUpdate()
      self.orderbook.timestamp = timestamp
      
      self.orderbook.asks = list(map(lambda i:
        ((md[pair + ':Binance:LinearPerpetual_ask_price_' + str(i)]).values[0], 
          (md[pair + ':Binance:LinearPerpetual_ask_vol_' + str(i)]).values[0]), range(10)))

      self.orderbook.bids =list(map(lambda i:
          ((md[pair + ':Binance:LinearPerpetual_bid_price_' + str(i)]).values[0], 
          (md[pair + ':Binance:LinearPerpetual_bid_vol_' + str(i)]).values[0]), range(10)))
      self.recieve_orderbook_latency = (-md['exchange_ts']+md['receive_ts_y']).values[0]
      
    time_to_next_tick: float #time until the next tick
    recieve_orderbook_latency: float
    recieve_trades_latency: list[float]
    orderbook: Optional[OrderbookSnapshotUpdate] = None
    trades: Optional[list[AnonTrade]] = None

position = []

class Strategy:
    def __init__(self, max_position: float, max_size: float, t_0: float) -> None:
        self.max_pos = max_position
        self.t_0 = t_0 * 10**6
        self.max_size = max_size

    def run(self, sim: "Sim"):
        while True:
            try:
                self.md_update, placed_orders, trades, disabled_orders, non_placed_orders, non_canceled_orders = sim.tick()
                self.counting_pos(trades)
                id_to_cancel = self.update_time_to_cancel()
                
                #PLACE ORDERS
                if ( abs(self.current_pos) <= self.max_pos):
                    if (bool(random.randint(0, 1))):
                        id = sim.place_order('ASK', random.uniform(1e-6, self.max_size), (self.md_update.orderbook.asks[0])[0])
                    else:
                        id = sim.place_order('BID', random.uniform(1e-6, self.max_size), (self.md_update.orderbook.bids[0])[0])
                    self.time_to_cancel.append((self.t_0, id))    
                elif ( self.current_pos < 0 ):
                    if (bool(random.randint(0, 1))):
                        id = sim.place_order('BID', random.uniform(1e-6, self.max_size), (self.md_update.orderbook.bids[0])[0])
                        self.time_to_cancel.append((self.t_0, id))  
                elif ( self.current_pos > 0 ):
                    if (bool(random.randint(0, 1))):
                        id = sim.place_order('ASK', random.uniform(1e-6, self.max_size), (self.md_update.orderbook.asks[0])[0])
                        self.time_to_cancel.append((self.t_0, id))  
                
                #CANCEL ORDERS 
                for id in id_to_cancel:
                    sim.cancel_order(id)
                  
                #call sim.place_order and sim.cancel_order here
                self.day_count += self.md_update.time_to_next_tick
                if (self.day_count >= 8.64e+8):
                    global position
                    self.day_count = 0 
                    count_pos = 0
                    for t in trades:
                        if (t.side == 'BID'):
                            count_pos += t.size*t.price
                        else:
                            count_pos -= t.size*t.price
                    position.append(count_pos)    
            except StopIteration:
                break
            
    def counting_pos(self, trades: list[OwnTrade]):
        pos = 0 
        for trade in trades:
            if (trade.side == 'ASK'):
                pos += trade.size
            if (trade.side == 'BID'):
                pos -= trade.size
        self.current_pos = pos
    
    def update_time_to_cancel(self):
        self.time_to_cancel = list(map(lambda x: (x[0]-self.md_update.time_to_next_tick, x[1]), self.time_to_cancel)) #!!!!!!
        id_to_cancel = list((map( lambda x: x[0], list(filter(lambda x: x[0] <= 0, self.time_to_cancel)))))
        self.time_to_cancel = list(filter(lambda x: x[0] > 0, self.time_to_cancel))
        return id_to_cancel
    
    day_count: float = 0 
    md_update: MdUpdate
    max_size: float
    max_pos: float
    current_pos: float
    t_0: float
    time_to_cancel: list[tuple[float, int]] = [] #(time, id)
    

def load_md_from_file(path: str) -> list[MdUpdate]:
    md = pd.read_csv(path)
    ts = np.unique(md['exchange_ts'])
    delta_t = np.diff(np.unique(md['exchange_ts'])).tolist()
    return list(map(lambda i: MdUpdate(md[md['exchange_ts'] == ts[i]], ts[i], delta_t[i]),
                range(10**3)))


class Sim:
    def __init__(self, execution_latency: float, md_latency: float) -> None:
        self.execution_latency = execution_latency * 10**6
        self.md_latency = md_latency * 10**6
        self.md = iter(load_md_from_file("./"+pair+"/md.csv"))
        
    #self.md - list of market data
    #self.current_md - current md
    def tick(self) -> MdUpdate:
        self.execute_orders()
        self.prepare_orders()
        self.prepare_trade()
        self.update_latency() 
        self.current_md = next(self.md)
        return self.current_md, self.placed_orders, self.trades, self.disabled_orders, self.not_placed_orders, self.not_canceled_orders

    def prepare_orders(self):
        
        temp_list = list(filter( lambda x: ( x[0] + self.current_md.recieve_orderbook_latency <= 0 ) and 
                               ( ( (x[1].side == 'ASK') and ( x[1].price >= (self.current_md.orderbook.asks[0])[0] ) ) or 
                                 ( (x[1].side == 'BID') and ( x[1].price <= (self.current_md.orderbook.bids[0])[0] ) ) ), 
                               self.not_placed_orders))
        self.placed_orders.extend(list(map(lambda x: x[1], temp_list))) #non placed orders -> placed
        
        
        self.disabled_orders.extend(list(map( lambda x: x[0], list(filter(lambda x: x[0] + self.current_md.recieve_orderbook_latency <= 0 and
                           not ( ( ( (x[1].side == 'ASK') and ( x[1].price >= (self.current_md.orderbook.asks[0])[0] ) ) or 
                                 ( (x[1].side == 'BID') and ( x[1].price <= (self.current_md.orderbook.bids[0])[0] ) ) ) ), 
                               self.not_placed_orders ))))) #list of id of disabled orders 
        self.not_placed_orders = list(filter(lambda x: not( x[1].id in self.disabled_orders),self.not_placed_orders))#clear
        #from disabled orders
        self.not_placed_orders = list(filter(lambda x: x[0] + self.current_md.recieve_orderbook_latency > 0, self.not_placed_orders)) #clear non placed from old orders
    
        temp_list_id = list(map( lambda x: x[1], list(filter(lambda x: x[0] + self.current_md.recieve_orderbook_latency <= 0, self.not_canceled_orders )))) #take old 'cancel'
        self.not_canceled_orders = list(filter(lambda x: x[0] + self.current_md.recieve_orderbook_latency > 0, self.not_canceled_orders)) #clear not canceled  
        for id in temp_list_id: #cancelling orders
            self.placed_orders = list(filter(lambda x: x.id != id, self.placed_orders))   
    
    def prepare_trade(self):
        self.trades.extend(list(map( lambda x: x[1], 
                                    list(filter(lambda x: x[0] <= 0, self.not_exec_trades ))))) #not_exec -> exec
        self.not_exec_trades = list(filter(lambda x: x[0] > 0, self.not_exec_trades ) ) #clear not_exec      
                                                                                                 
    def execute_orders(self): #creating trade 
        for order in self.placed_orders:
            anon_trades = self.current_md.trades
            count = 0 
            for trade in anon_trades:
                if  ((order.side == 'ASK' and trade.price >= order.price) or 
                     ( order.side == 'BID' and trade.price <= order.price)):
                    new_trade = OwnTrade()                        
                    self.trade_id += 1
                    new_trade.timestamp = trade.timestamp
                    new_trade.trade_id = self.trade_id 
                    new_trade.order_id = order.id 
                    new_trade.side = trade.side
                    new_trade.size = order.size
                    new_trade.price = trade.price
                    
                    self.not_exec_trades.append((self.md_latency+self.current_md.recieve_trades_latency[count]
                                                 ,new_trade))
                    
                count += 1
            
    def place_order(self, side: str, size: float, price: float):
        order = Order()
        self.order_id += 1 
        order.id = self.order_id
        order.side = side
        order.size = size
        order.price = price
        self.not_placed_orders.append((self.execution_latency, order))
        return order.id
              
    def cancel_order(self, id: int):
        self.not_canceled_orders.append((self.execution_latency, id))
        pass
    
    def update_latency(self):
        self.not_placed_orders = list(map(lambda x: (x[0]-self.current_md.time_to_next_tick, x[1]), self.not_placed_orders))
        self.not_exec_trades = list(map(lambda x: (x[0]-self.current_md.time_to_next_tick, x[1]), self.not_exec_trades))
        self.not_canceled_orders = list(map(lambda x: (x[0]-self.current_md.time_to_next_tick, x[1]), self.not_canceled_orders))

    order_id: int = -1
    trade_id: int = -1
    execution_latency: float 
    md_latency: float
    current_md: MdUpdate
    
    not_placed_orders: list[tuple[float, Order]] = [] #ордер отправлен, еще не дошел до биржи
    not_canceled_orders: list[tuple[float, float]] = []
    not_exec_trades: list[tuple[float, OwnTrade]] = [] #Создали трейд, еще не отправили в стратегию 
    trades: list[OwnTrade] = []
    disabled_orders: list[int]= []
    placed_orders: list[Order] = [] 


if __name__ == "__main__":
    strategy = Strategy(10, 0.5, 100)
    sim = Sim(10, 10)
    strategy.run(sim)
    plt.plot(np.array(position))